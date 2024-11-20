import logging
import os
from dataclasses import dataclass

import requests
import torch
from esm import FastaBatchedDataset, pretrained

logger = logging.getLogger(__name__)


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


def write_sequence_from_ensembl(ensembl_gene_id: list[str], fasta_file_output: str):
    assert fasta_file_output.endswith(".fasta"), "Output file must be in FASTA format"
    missing_ids = []
    results = {}
    with open(fasta_file_output, "w") as f:
        for gene_id in ensembl_gene_id:
            canonical_transcript_info = fetch_canonical_transcript_info(gene_id)
            if canonical_transcript_info:
                transcript_id = canonical_transcript_info["transcript_id"]
                display_name = canonical_transcript_info["display_name"]
                protein_sequence = fetch_protein_sequence(transcript_id)
            if protein_sequence:
                results[gene_id] = protein_sequence
                f.write(
                    f">query_{ensembl_gene_id}_canonical_transcript_{transcript_id}_Name_{display_name}\n"
                )
                f.write(f"{protein_sequence}\n")
            else:
                missing_ids.append(gene_id)
    print(f"Missing sequence for ids: {set(missing_ids)}")
    return results


@dataclass
class EmbeddingConfig:
    model_name: str
    output_dir: str
    include: str = "mean"
    use_gpu: bool = True
    toks_per_batch: int = 4096
    truncation_seq_length: int = 1022
    repr_layers: list[int] = [-1]
    _valid_includes = ["per_tok", "mean", "bos"]

    def __post_init__(self):
        self.fasta_path = os.path.join(self.output_dir, "sequences.fasta")
        assert (
            self.include in self._valid_includes
        ), f"Must be one of {self._valid_includes}"


def embedding_from_seq(config: EmbeddingConfig):
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
            print(
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
                if "per_tok" in config.include:
                    result["representations"] = {
                        layer: t[i, 1 : truncate_len + 1].clone()
                        for layer, t in representations.items()
                    }
                if "mean" in config.include:
                    result["mean_representations"] = {
                        layer: t[i, 1 : truncate_len + 1].mean(0).clone()
                        for layer, t in representations.items()
                    }
                if "bos" in config.include:
                    result["bos_representations"] = {
                        layer: t[i, 0].clone() for layer, t in representations.items()
                    }

                torch.save(
                    result,
                    output_file,
                )
