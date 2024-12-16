import os
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property

import anndata as ad
import pandas as pd
import requests
from transformers import AutoTokenizer, EsmModel

from cfp._logging import logger

try:
    import torch
    from transformers import AutoTokenizer, EsmModel
except ImportError as e:
    torch = None
    AutoTokenizer = None
    EsmModel = None
    raise ImportError(
        "To use gene embedding, please install `fair-esm` and `torch` \
            e.g. via `pip install -e .['embedding']`."
    ) from e


def fetch_canonical_transcript_info(ensembl_gene_id: str) -> dict[str, str] | None:
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


def fetch_protein_sequence(ensembl_transcript_id: str) -> str:
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


def prot_sequence_from_ensembl(ensembl_gene_id: list[str]) -> pd.DataFrame:
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
        df_iter = pd.DataFrame(data, columns=columns)
        df = pd.concat([df, df_iter])

    if missing_ids:
        logger.info(f"Missing sequence for ids: {set(missing_ids)}")
    return df


def order_to_batch_list(unordered_list, batch_idx):
    all_batch_names = []
    for batch in batch_idx:
        batch_names = [unordered_list[i] for i in batch]
        all_batch_names.append(batch_names)
    return all_batch_names


class BatchedDataset:
    """Modified batched dataset from fair-esm `c9c7d4f0fec964ce10c3e11dccec6c16edaa5144`"""

    def __init__(self, sequence_labels, sequence_strs):
        self.sequence_labels = list(sequence_labels)
        self.sequence_strs = list(sequence_strs)

    def __len__(self):
        return len(self.sequence_labels)

    def __getitem__(self, idx):
        return self.sequence_labels[idx], self.sequence_strs[idx]

    def get_batch_indices(self, toks_per_batch, extra_toks_per_seq=0):
        sizes = [(len(s), i) for i, s in enumerate(self.sequence_strs)]
        sizes.sort()
        batches = []
        buf = []
        max_len = 0

        def _flush_current_buf():
            nonlocal max_len, buf
            if len(buf) == 0:
                return
            batches.append(buf)
            buf = []
            max_len = 0

        for sz, i in sizes:
            sz += extra_toks_per_seq
            if max(sz, max_len) * (len(buf) + 1) > toks_per_batch:
                _flush_current_buf()
            max_len = max(max_len, sz)
            buf.append(i)

        _flush_current_buf()
        return batches


def create_dataloader(prot_names, sequences, toks_per_batch, collate_fn):
    dataset = BatchedDataset(prot_names, sequences)
    batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_sampler=batches,
    )
    return data_loader


def _get_esm_collate_fn(tokenizer, max_length, truncation, return_tensors):
    def collate_fn(batch):
        # batch of tuples (gene_id, sequence)
        gene_id, seq = zip(*batch, strict=False)
        metadata = {"gene_id": gene_id, "protein_sequence": seq}
        token = tokenizer(
            seq,
            padding=True,
            max_length=max_length,
            truncation=truncation,
            return_tensors=return_tensors,
        )
        return metadata, token

    return collate_fn


def get_model_and_tokenizer(model_name, use_cuda):
    model_path = os.path.join("facebook", model_name)
    model = EsmModel.from_pretrained(model_path)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if use_cuda:
        model = model.cuda()
    model.requires_grad_(False)
    return model, tokenizer


def get_esm_embedding(
    adata: ad.AnnData,
    gene_key: str | Iterable[str] = "gene_target_",
    null_value: str | None = None,
    gene_emb_key: str = "gene_embedding",
    copy: bool = False,
    esm_model_name: str = "esm2_t36_3B_UR50D",
    toks_per_batch: int = 4096,
    trunc_len: int | None = 1022,
    truncation: bool = True,
    use_cuda: bool = True,
) -> ad.AnnData | None:
    """
    Create gene embeddings from adata object using ESM2 model :cite:`lin:2023`.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    gene_key : str | Iterable[str]
        prefix in `adata.obs` containing gene names or list of keys.
    null_value : str | None
        Value to ignore (useful when using combinations of KO).
    gene_emb_key : str
        Key to store gene embeddings in `adata.uns`.
    copy : bool
        Return a copy of `adata` instead of updating it in place.
    esm_model_name : str
        Name of the ESM model to use.
    toks_per_batch : int
        Number of tokens per batch.
    trunc_len : int | None
        Maximum length of the sequence.
    truncation : bool
        Whether to truncate the sequence.
    use_cuda : bool
        Use GPU if available.

    Returns
    -------
    AnnData
        If `copy` is :obj:`True`, returns a new :class:`~anndata.AnnData`
        Sets the following fields:
        `adata.uns[gene_emb_key]`: Gene embeddings.
        `adata.uns[gene_emb_key + "_metadata"]`: Metadata for gene embeddings.
    """
    if copy:
        adata = adata.copy()
    if isinstance(gene_key, str):
        # We use it as a prefix
        mask_col = adata.obs.columns.str.startswith(gene_key)
        columns = adata.obs.columns[mask_col]
    else:
        columns = gene_key
    unique_genes = set()
    for col in columns:
        unique_genes.update(adata.obs[col].unique().tolist())
    unique_genes = unique_genes - {null_value, None}
    metadata = prot_sequence_from_ensembl(unique_genes)
    to_emb = metadata[metadata.protein_sequence.notnull()]
    use_cuda = use_cuda and torch.cuda.is_available()
    esm, tokenizer = get_model_and_tokenizer(esm_model_name, use_cuda)
    data_loader = create_dataloader(
        prot_names=to_emb["gene_id"].to_list(),
        sequences=to_emb["protein_sequence"].to_list(),
        toks_per_batch=toks_per_batch,
        collate_fn=_get_esm_collate_fn(
            tokenizer, max_length=trunc_len, truncation=truncation, return_tensors="pt"
        ),
    )
    results = {}
    for batch_metadata, batch in data_loader:
        if use_cuda:
            batch = {k: v.cuda() for k, v in batch.items()}
        batch_results = esm(**batch).last_hidden_state
        for i, name in enumerate(batch_metadata["gene_id"]):
            trunc_len = min(trunc_len, len(batch_metadata["protein_sequence"][i]))
            emb = batch_results[i, 1 : trunc_len + 1].mean(0).clone()
            results[name] = emb
    adata.uns[gene_emb_key] = results
    adata.uns[gene_emb_key + "_metadata"] = metadata
    if copy:
        return adata
