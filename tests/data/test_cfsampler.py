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

        assert "src_lin" in sample_1
        assert "tgt_lin" in sample_1
        assert "src_condition" in sample_1
        for k, v in sample_1.items():
            assert v.shape[0] == batch_size
            assert v.shape[1:] == sample_2[k].shape[1:]
