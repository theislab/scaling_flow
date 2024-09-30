from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from scvi import REGISTRY_KEYS
from scvi.distributions import JaxNegativeBinomialMeanDisp as NegativeBinomial
from scvi.module._jaxvae import JaxVAE
from scvi.module.base import LossOutput, flax_configure


def _get_dict_if_none(param):
    param = {} if not isinstance(param, dict) else param

    return param


@flax_configure
class CFJaxVAE(JaxVAE):
    """Variational autoencoder model."""

    n_input: int
    n_batch: int
    gene_likelihood: str
    n_hidden: int = 128
    n_latent: int = 30
    dropout_rate: float = 0.0
    n_layers: int = 1
    eps: float = 1e-8
    training: bool = True

    @property
    def required_rngs(self):
        return ("params", "dropout", "z")

    def _get_inference_input(self, tensors: dict[str, jnp.ndarray]):
        """Get input for inference."""
        x = tensors[REGISTRY_KEYS.X_KEY]

        input_dict = {"x": x}
        return input_dict

    def inference(self, x: jnp.ndarray, n_samples: int = 1) -> dict[str, jnp.ndarray]:
        """Run inference model."""
        mean, var = self.encoder(x, training=self.training)
        stddev = jnp.sqrt(var) + self.eps

        qz = dist.Normal(mean, stddev)
        z_rng = self.make_rng("z")
        sample_shape = () if n_samples == 1 else (n_samples,)
        z = qz.rsample(z_rng, sample_shape=sample_shape)

        return {"qz": qz, "z": z}

    def _get_generative_input(
        self,
        tensors: dict[str, jnp.ndarray],
        inference_outputs: dict[str, jnp.ndarray],
    ):
        """Get input for generative model."""
        x = tensors[REGISTRY_KEYS.X_KEY]
        z = inference_outputs["z"]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]

        input_dict = {
            "x": x,
            "z": z,
            "batch_index": batch_index,
        }
        return input_dict

    def generative(self, x, z, batch_index) -> dict[str, jnp.ndarray]:
        """Run generative model."""
        # one hot adds an extra dimension
        batch = jax.nn.one_hot(batch_index, self.n_batch).squeeze(-2)
        rho_unnorm, disp = self.decoder(z, batch, training=self.training)
        disp_ = jnp.exp(disp)
        rho = jax.nn.softmax(rho_unnorm, axis=-1)
        total_count = x.sum(-1)[:, jnp.newaxis]
        mu = total_count * rho

        if self.gene_likelihood == "nb":
            disp_ = jnp.exp(disp)
            px = NegativeBinomial(mean=mu, inverse_dispersion=disp_)
        elif self.gene_likelihood == "poisson":
            px = dist.Poisson(mu)
        elif self.gene_likelihood == "normal":
            px = rho
        else:
            raise NotImplementedError(
                f"Invalid gene_likelihood {self.gene_likelihood}."
            )
        return {"px": px, "rho": rho}

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1.0,
    ):
        """Compute loss."""
        x = tensors[REGISTRY_KEYS.X_KEY]
        px = generative_outputs["px"]
        qz = inference_outputs["qz"]
        if self.gene_likelihood == "normal":
            reconst_loss = jnp.sum((px - x) ** 2, axis=-1)
        else:
            reconst_loss = -px.log_prob(x).sum(-1)
        kl_divergence_z = dist.kl_divergence(qz, dist.Normal(0, 1)).sum(-1)

        kl_local_for_warmup = kl_divergence_z
        weighted_kl_local = kl_weight * kl_local_for_warmup

        loss = jnp.mean(reconst_loss + weighted_kl_local)

        kl_local = kl_divergence_z
        return LossOutput(
            loss=loss, reconstruction_loss=reconst_loss, kl_local=kl_local
        )

    def get_jit_generative_fn_old(
        self,
        get_generative_input_kwargs: dict[str, Any] | None = None,
        generative_kwargs: dict[str, Any] | None = None,
    ) -> Callable[
        [dict[str, jnp.ndarray], dict[str, jnp.ndarray]], dict[str, jnp.ndarray]
    ]:
        """
        Create a method to run generative using the bound module.

        Parameters
        ----------
        get_generative_input_kwargs
            Keyword arguments to pass to subclass `_get_generative_input`
        generative_kwargs
            Keyword arguments  for subclass `generative` method

        Returns
        -------
        A callable taking rngs and array_dict as input and returning the output
        of the `inference` method. This callable runs `_get_inference_input`.
        """
        vars_in = {"params": self.params, **self.state}
        get_generative_input_kwargs = _get_dict_if_none(get_generative_input_kwargs)
        generative_kwargs = _get_dict_if_none(generative_kwargs)

        # @jax.jit TODO: check why jitting fails
        def _run_generative(rngs, array_dict, inference_outputs):
            module = self.clone()
            generative_input = module._get_generative_input(
                array_dict, inference_outputs
            )
            out = module.apply(
                vars_in,
                rngs=rngs,
                method=module.generative,
                mutable=False,
                **generative_input,
                **generative_kwargs,
            )
            return out

        return _run_generative  # type: ignore[return-value]

    def get_jit_generative_fn(
        self,
        get_generative_input_kwargs: dict[str, Any] | None = None,
        generative_kwargs: dict[str, Any] | None = None,
    ):

        vars_in = {"params": self.params, **self.state}
        get_generative_input_kwargs = _get_dict_if_none(get_generative_input_kwargs)
        generative_kwargs = _get_dict_if_none(generative_kwargs)

        # @jax.jit
        def _run_generative(rngs, array_dict, inference_outputs):
            module = self.clone()
            generative_input = module._get_generative_input(
                array_dict, inference_outputs
            )
            out = module.apply(
                vars_in,
                rngs=rngs,
                method=module.generative,
                mutable=False,
                **generative_input,
                **generative_kwargs,
            )
            return out

        return _run_generative
