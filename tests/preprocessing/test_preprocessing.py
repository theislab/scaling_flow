import anndata as ad
import numpy as np
import pytest


class TestPreprocessing:
    @pytest.mark.parametrize(
        "query_key_and_type", [("compound_name", "name"), ("compound_cid", "cid")]
    )
    def test_annotate_compounds(
        self, adata_with_compounds: ad.AnnData, query_key_and_type
    ):
        import cfp

        prefix = "compound"

        cfp.pp.annotate_compounds(
            adata_with_compounds,
            query_keys=query_key_and_type[0],
            query_id_type=query_key_and_type[1],
            obs_key_prefixes=[prefix],
            copy=False,
        )
        assert f"{prefix}_pubchem_name" in adata_with_compounds.obs
        assert f"{prefix}_pubchem_ID" in adata_with_compounds.obs
        assert f"{prefix}_smiles" in adata_with_compounds.obs

    @pytest.mark.parametrize("n_bits", [512, 1024])
    def test_get_molecular_fingerprints(self, adata_with_compounds: ad.AnnData, n_bits):
        import cfp

        uns_key_added = "compound_fingerprints"

        cfp.pp.get_molecular_fingerprints(
            adata_with_compounds,
            compound_key="compound_name",
            smiles_key="compound_smiles",
            uns_key_added=uns_key_added,
            n_bits=n_bits,
            copy=False,
        )

        assert uns_key_added in adata_with_compounds.uns
        assert (
            next(iter(adata_with_compounds.uns[uns_key_added].values())).shape[0]
            == n_bits
        )

    @pytest.mark.parametrize("uns_key_added", ["compounds", "compounds_onehot"])
    @pytest.mark.parametrize("exclude_values", [None, "GW0742"])
    def test_encode_onehot(
        self, adata_with_compounds: ad.AnnData, uns_key_added, exclude_values
    ):
        import cfp

        cfp.pp.encode_onehot(
            adata_with_compounds,
            covariate_keys="compound_name",
            uns_key_added=uns_key_added,
            exclude_values=exclude_values,
            copy=False,
        )

        if exclude_values is None:
            expected_compounds = adata_with_compounds.obs["compound_name"].unique()
        else:
            expected_compounds = np.setdiff1d(
                adata_with_compounds.obs["compound_name"].unique(), exclude_values
            )
        assert uns_key_added in adata_with_compounds.uns
        assert len(adata_with_compounds.uns[uns_key_added]) == len(expected_compounds)
        assert adata_with_compounds.uns[uns_key_added][expected_compounds[0]].shape[
            0
        ] == len(expected_compounds)
        assert np.all(
            [c in adata_with_compounds.uns[uns_key_added] for c in expected_compounds]
        )
