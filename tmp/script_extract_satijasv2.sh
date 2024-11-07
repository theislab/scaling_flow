# bash script_extract_satijas.sh

DATA_FILE=/network/scratch/g/guillaume.huguet/projects/esm/perturb_emb/satijas_v2/satijas_protein_sequences.fasta
OUTPUT_DIR=/network/scratch/g/guillaume.huguet/projects/esm/perturb_emb/satijas_v2

python scripts/extract.py esm2_t36_3B_UR50D $DATA_FILE $OUTPUT_DIR --include mean