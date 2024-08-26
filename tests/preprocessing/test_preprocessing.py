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
