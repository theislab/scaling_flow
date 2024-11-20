import requests
import scanpy as sc


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


def augment_adata_with_gene_embeddings(
    adata: sc.AnnData, treat_key: str = "perturbation"
) -> sc.AnnData:
    gene_ids = adata.obs[treat_key].unique().tolist()
    res = write_sequence_from_ensembl(gene_ids, "gene_embeddings.fasta")
    # TODO: not sure how / where to add them.
    adata.obs["gene_embedding"] = res
    pass
