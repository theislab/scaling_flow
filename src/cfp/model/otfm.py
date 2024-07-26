from collections.abc import Callable
from typing import Any

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
from flax.training import train_state
from ott.neural.methods.flows import dynamics
from ott.solvers import utils as solver_utils

from cfp._types import ArrayLike
from cfp.networks.velocity_field import ConditionalVelocityField

__all__ = ["OTFlowMatching"]


class OTFlowMatching:
    """(Optimal transport) flow matching :cite:`lipman:22`.

    With an extension to OT-FM :cite:`tong:23,pooladian:23`.

    Parameters
    ----------
      vf: Vector field parameterized by a neural network.
      flow: Flow between the source and the target distributions.
      match_fn: Function to match samples from the source and the target
        distributions. It has a ``(src, tgt) -> matching`` signature.
      time_sampler: Time sampler with a ``(rng, n_samples) -> time`` signature.
    """

    def __init__(
        self,
        vf: ConditionalVelocityField,
        flow: dynamics.BaseFlow,
        match_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] | None = None,
        time_sampler: Callable[
            [jax.Array, int], jnp.ndarray
        ] = solver_utils.uniform_sampler,
        **kwargs: Any,
    ):
        self.vf = vf
        self.flow = flow
        self.time_sampler = time_sampler
        self.match_fn = match_fn

        self.vf_state = self.vf.create_train_state(
            input_dim=self.vf.output_dims[-1], **kwargs
        )
        self.step_fn = self._get_step_fn()

    def _get_step_fn(self) -> Callable:

        @jax.jit
        def step_fn(
            rng: jax.Array,
            vf_state: train_state.TrainState,
            source: jnp.ndarray,
            target: jnp.ndarray,
            source_conditions: jnp.ndarray | None,
        ) -> tuple[Any, Any]:

            def loss_fn(
                params: jnp.ndarray,
                t: jnp.ndarray,
                source: jnp.ndarray,
                target: jnp.ndarray,
                source_conditions: jnp.ndarray | None,
                rng: jax.Array,
            ) -> jnp.ndarray:
                rng_flow, rng_dropout = jax.random.split(rng, 2)
                x_t = self.flow.compute_xt(rng_flow, t, source, target)
                v_t = vf_state.apply_fn(
                    {"params": params},
                    t,
                    x_t,
                    source_conditions,
                    rngs={"dropout": rng_dropout},
                )
                u_t = self.flow.compute_ut(t, source, target)

                return jnp.mean((v_t - u_t) ** 2)

            batch_size = len(source)
            key_t, key_model = jax.random.split(rng, 2)
            t = self.time_sampler(key_t, batch_size)
            grad_fn = jax.value_and_grad(loss_fn)
            loss, grads = grad_fn(
                vf_state.params, t, source, target, source_conditions, key_model
            )
            return vf_state.apply_gradients(grads=grads), loss

        return step_fn

    def outer_step_fn(
        self,
        rng: jnp.ndarray,
        batch: dict[str, ArrayLike],
    ):
        """Outer step function for OTFM solver."""
        src, tgt = batch["src_cell_data"], batch["tgt_cell_data"]
        condition = batch.get("condition")
        rng_resample, rng_step_fn = jax.random.split(rng, 2)
        if self.match_fn is not None:
            tmat = self.match_fn(src, tgt)
            src_ixs, tgt_ixs = solver_utils.sample_joint(rng_resample, tmat)
            src, tgt = src[src_ixs], tgt[tgt_ixs]

        self.vf_state, loss = self.step_fn(
            rng_step_fn,
            self.vf_state,
            src,
            tgt,
            condition,
        )
        return loss

    def get_condition_embedding(self, condition: ArrayLike) -> ArrayLike:
        """Encode conditions

        Args:
            condition: Conditions to encode

        Returns
        -------
            Encoded conditions
        """
        cond_embed = self.vf.apply(
            {"params": self.vf_state.params},
            condition,
            method="get_condition_embedding",
        )
        return np.asarray(cond_embed)

    def predict(self, x: ArrayLike, condition: ArrayLike, **kwargs: Any) -> ArrayLike:
        """Predict the push-forward of the data.

        TODO: add the option to return the ODE solution

        Parameters
        ----------
        x : ArrayLike
            Input data of shape [batch_size, ...].
        condition : ArrayLike
            Condition of the input data of shape [batch_size, ...].
        kwargs : Any
            Keyword arguments for the ODE solver.

        Returns
        -------
        The generated single cells.
        """
        kwargs.setdefault("dt0", None)
        kwargs.setdefault("solver", diffrax.Tsit5())
        kwargs.setdefault(
            "stepsize_controller", diffrax.PIDController(rtol=1e-5, atol=1e-5)
        )

        def vf(
            t: jnp.ndarray, x: jnp.ndarray, cond: dict[str, jnp.ndarray] | None
        ) -> jnp.ndarray:
            params = self.model.vf_state.params
            return self.model.vf_state.apply_fn(
                {"params": params}, t, x, cond, train=False
            )

        def solve_ode(x: jnp.ndarray, cond: jnp.ndarray | None) -> jnp.ndarray:
            ode_term = diffrax.ODETerm(vf)
            result = diffrax.diffeqsolve(
                ode_term,
                t0=0.0,
                t1=1.0,
                y0=x,
                args=cond,
                dt0=None,
                solver=diffrax.Tsit5(),
                stepsize_controller=diffrax.PIDController(rtol=1e-5, atol=1e-5),
            )
            return result.ys[0]

        x_pred = jax.jit(jax.vmap(solve_ode, in_axes=[0, None]))(x, condition)
        return np.array(x_pred)
