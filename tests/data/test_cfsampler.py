import jax
import pytest

from cfp.data.dataloader import CFSampler


class TestCFSampler:
    @pytest.mark.parametrize("batch_size", [1, 31])
    def test_sampling_no_combinations(self, pdata, batch_size):
        sampler = CFSampler(data=pdata, batch_size=batch_size)
        rng_1 = jax.random.PRNGKey(0)
        rng_2 = jax.random.PRNGKey(1)

        sample_1 = sampler.sample(rng_1)
        sample_2 = sampler.sample(rng_2)

        assert "src_cell_data" in sample_1
        assert "tgt_cell_data" in sample_1
        assert "src_condition" in sample_1
        assert sample_1["src_cell_data"].shape[0] == batch_size
        assert sample_2["src_cell_data"].shape[0] == batch_size
        assert sample_1["tgt_cell_data"].shape[0] == batch_size
        assert sample_2["tgt_cell_data"].shape[0] == batch_size
        assert sample_1["src_condition"]["dosage"].shape[0] == batch_size
        assert sample_2["src_condition"]["dosage"].shape[0] == batch_size
