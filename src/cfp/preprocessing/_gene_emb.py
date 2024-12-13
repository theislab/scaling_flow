import os
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property

import anndata as ad
import pandas as pd
import requests

from cfp._logging import logger

try:
    import torch
    from esm import FastaBatchedDataset, pretrained
except ImportError as e:
    torch = None
    FastaBatchedDataset = None
    pretrained = None
    raise ImportError(
        "To use gene embedding, please install `fair-esm` and `torch` \
            e.g. via `pip install -e .['embedding']`."
    ) from e


def fetch_canonical_transcript_info(ensembl_gene_id: str):
    server = "https://rest.ensembl.org"
    ext = f"/lookup/id/{ensembl_gene_id}?expand=1"
    headers = {"Content-Type": "application/json"}

    # Fetch gene information
    response = requests.get(server + ext, headers=headers)
    if not response.ok:
        response.raise_for_status()

    gene_data = response.json()
    transcripts = gene_data.get("Transcript", [])

    # Find the canonical transcript
    canonical_transcript_info = None
    for transcript in transcripts:
        if transcript.get("is_canonical"):
            canonical_transcript_info = {
                "transcript_id": transcript["id"],
                "display_name": transcript.get("display_name", "Unknown Protein"),
                "biotype": transcript.get("biotype", "Unknown Biotype"),
            }
            break

    return canonical_transcript_info


def fetch_protein_sequence(ensembl_transcript_id: str):
    server = "https://rest.ensembl.org"
    ext = f"/sequence/id/{ensembl_transcript_id}?type=protein"
    headers = {"Content-Type": "application/json"}

    response = requests.get(server + ext, headers=headers)
    if not response.ok:
        response.raise_for_status()

    protein_data = response.json()
    return protein_data.get("seq", "")


@dataclass
class GeneInfo:
    gene_id: str

    def __post_init__(self):
        self._is_protein_coding: bool = False
        self.transcript_id: str | None = None
        self.display_name: str | None = None
        self.canonical_transcript_info = fetch_canonical_transcript_info(self.gene_id)
        if self.canonical_transcript_info:
            self.transcript_id = self.canonical_transcript_info["transcript_id"]
            self.display_name = self.canonical_transcript_info["display_name"]
            self._is_protein_coding = (
                self.canonical_transcript_info["biotype"] == "protein_coding"
            )

    @property
    def is_protein_coding(self) -> bool:
        return self._is_protein_coding

    @cached_property
    def protein_sequence(self) -> str | None:
        if self.is_protein_coding:
            return fetch_protein_sequence(self.transcript_id)
        return None

    @property
    def seq_len(self) -> int | None:
        if self.protein_sequence:
            return len(self.protein_sequence)
        return None


def write_sequence_from_ensembl(ensembl_gene_id: list[str], fasta_file_output: str):
    assert fasta_file_output.endswith(".fasta"), "Output file must be in FASTA format"
    missing_ids = []
    results = {}
    columns = [
        "gene_id",
        "transcript_id",
        "display_name",
        "is_protein_coding",
        "seq_len",
        "protein_sequence",
    ]
    df = pd.DataFrame(columns=columns)
    with open(fasta_file_output, "w") as f:
        for gene_id in ensembl_gene_id:
            gene_info = GeneInfo(gene_id)
            results[gene_id] = gene_info.protein_sequence
            data = [
                [
                    gene_id,
                    gene_info.transcript_id,
                    gene_info.display_name,
                    gene_info.is_protein_coding,
                    gene_info.seq_len,
                    gene_info.protein_sequence,
                ]
            ]
            if gene_info.is_protein_coding:
                f.write(f">{gene_id}\n")
                f.write(f"{gene_info.protein_sequence}\n")
            else:
                missing_ids.append(gene_id)
            df_iter = pd.DataFrame(data, columns=columns)
            df = pd.concat([df, df_iter])

    if missing_ids:
        logger.info(f"Missing sequence for ids: {set(missing_ids)}")
    df.to_csv(
        os.path.join(os.path.dirname(fasta_file_output), "gene_info.csv"), index=False
    )
    return df


@dataclass
class EmbeddingConfig:
    model_name: str = "esm2_t36_3B_UR50D"
    output_dir: str = "gene_embeddings"
    use_gpu: bool = True
    toks_per_batch: int = 4096
    truncation_seq_length: int = 1022
    repr_layers: list[int] | None = None

    def __post_init__(self):
        self.fasta_path = os.path.join(self.output_dir, "sequences.fasta")
        if self.repr_layers is None:
            self.repr_layers = [-1]


def embedding_from_seq(config: EmbeddingConfig, save_to_disk: bool = False):
    model, alphabet = pretrained.load_model_and_alphabet(config.model_name)
    model.eval()
    if torch.cuda.is_available() and config.use_gpu:
        model = model.cuda()
    dataset = FastaBatchedDataset.from_file(config.fasta_path)
    batches = dataset.get_batch_indices(config.toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=alphabet.get_batch_converter(config.truncation_seq_length),
        batch_sampler=batches,
    )
    logger.info(f"Read {len(dataset)} sequences from {config.fasta_path}")
    results: dict[str, dict[str, torch.Tensor]] = {}
    # Don't overwrite existing embeddings
    for file in os.listdir(config.output_dir):
        if file.endswith(".pth"):
            logger.info(
                f"Found existing .pth file in {config.output_dir}. Skipping embedding generation."
            )
            return

        assert all(
            -(model.num_layers + 1) <= i <= model.num_layers for i in config.repr_layers
        )
    repr_layers = [
        (i + model.num_layers + 1) % (model.num_layers + 1) for i in config.repr_layers
    ]

    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            logger.info(
                f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)"
            )
            if torch.cuda.is_available() and config.use_gpu:
                toks = toks.to(device="cuda", non_blocking=True)

            out = model(toks, repr_layers=repr_layers, return_contacts=False)
            representations = {
                layer: t.to(device="cpu") for layer, t in out["representations"].items()
            }

            for i, label in enumerate(labels):
                output_file = os.path.join(config.output_dir, f"{label}.pth")
                result = {"label": label}
                truncate_len = min(config.truncation_seq_length, len(strs[i]))
                # Call clone on tensors to ensure tensors are not views into a larger representation
                # See https://github.com/pytorch/pytorch/issues/1995
                emb = {
                    layer: t[i, 1 : truncate_len + 1].mean(0).clone()
                    for layer, t in representations.items()
                }
                average_layers = torch.stack(list(emb.values())).mean(0)
                result["embedding"] = average_layers
                if save_to_disk:
                    torch.save(
                        result,
                        output_file,
                    )
                    results[label] = output_file
                else:
                    results[label] = average_layers
    return results


def create_gene_embedding(
    adata: ad.AnnData,
    gene_key: str | Iterable[str],
    null_vallue: str | None = None,
    embedding_config: EmbeddingConfig | None = None,
    gene_emb_key: str = "gene_embedding",
    save_to_disk: bool = False,
):
    adata = adata.copy()
    if embedding_config is None:
        embedding_config = EmbeddingConfig(
            model_name="esm2_t36_3B_UR50D",
            output_dir="gene_embeddings" if save_to_disk else "tmp_fasta",
            use_gpu=True,  # Use GPU only if available
            toks_per_batch=4096,
            truncation_seq_length=1022,
            repr_layers=[-1],
        )
    os.makedirs(embedding_config.output_dir, exist_ok=True)
    if isinstance(gene_key, str):
        # We use it as a prefix
        mask_col = adata.obs.columns.str.startswith(gene_key)
        columns = adata.obs.columns[mask_col]
    else:
        columns = gene_key
    unique_genes = set()
    for col in columns:
        unique_genes.update(adata.obs[col].unique().tolist())
    unique_genes = unique_genes - {null_vallue, None}
    print(unique_genes)
    metadata = write_sequence_from_ensembl(unique_genes, embedding_config.fasta_path)
    results = embedding_from_seq(embedding_config)
    adata.uns[gene_emb_key] = results
    adata.uns[gene_emb_key + "_metadata"] = metadata
    if not save_to_disk:
        import shutil

        shutil.rmtree(embedding_config.output_dir)
    return adata