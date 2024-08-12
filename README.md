[![Python version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)]()
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# CellFlow

Modeling complex perturbations with flow matching and optimal transport

## Quick start ⚡️

```python
import cfp

# Initialize CellFlow
cf = cfp.model.CellFlow(adata, solver="otfm")

# Prepare the training data and perturbation conditions
cf.prepare_data(
    sample_rep="X_pca",
    control_key="CTRL",
    perturbation_covariates={
        "drugs": ["Dabrafenib", "Trametinib"],
        "times": ["Dabrafenib_time", "Trametinib_time"],
    },
    perturbation_covariate_reps={
        "drugs": "drug_embeddings",
    },
    sample_covariates=["cell_line"],
    sample_covariate_reps={
        "cell_line": "cell_line_embeddings",
    },
)

# Prepare the model
cf.prepare_model(
    encode_conditions=True,
    condition_embedding_dim=32,
    hidden_dims=(128, 128),
    decoder_dims=(128, 128),
)

# Train the model
cf.train(
    num_iterations=1000,
    batch_size=128,
)

# Make predictions
X_pca_pred = cf.predict(
    adata_ctrl,
    condition_data=test_condition_df,
)

# Get condition embeddings
condition_embeddings = cf.get_condition_embeddings(adata)
```
