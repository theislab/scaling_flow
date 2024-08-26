import anndata as ad
import numpy as np
import pytest


class TestPreprocessing:
    @pytest.mark.parametrize(
        "query_id_and_type", [("compound_name", "name"), ("compound_cid", "cid")]
    )
    def test_annotate_compounds(
        self, adata_with_compounds: ad.AnnData, query_id_and_type
    ):
        import cfp

        cfp.pp.annotate_compounds(
            adata_with_compounds,
            query_id=query_id_and_type[0],
            query_id_type=query_id_and_type[1],
            copy=False,
        )
        assert "pubchem_name" in adata_with_compounds.obs
        assert "pubchem_ID" in adata_with_compounds.obs
        assert "smiles" in adata_with_compounds.obs

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
