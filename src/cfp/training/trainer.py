import functools
import os
import sys
import traceback
from typing import Literal, Optional, Dict, Callable, Iterable, Union, Any
from functools import partial

import hydra
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import optax
import orbax
import scanpy as sc
import wandb
from omegaconf import DictConfig, OmegaConf
from ott.neural import datasets
from ott.neural.methods.flows import dynamics, otfm, genot
from ott.neural.networks.layers import time_encoder
from ott.solvers import utils as solver_utils
from torch.utils.data import DataLoader
from tqdm import tqdm

from cfp.metrics import compute_mean_metrics, compute_metrics, compute_metrics_fast


class CFP:

    def __init__(self, model: Union[otfm.OTFlowmatching, genot.GENOT], dl: Iterable):
        self.model = model
        self.dl = dl

    def train(self, num_iterations: int, valid_freq: int, callback_fn: Callable[[Union[otfm.OTFlowmatching, genot.GENOT]], Any]) -> None:

        training_logs = {"loss": []}
        rng = jax.random.PRNGKey(0)
        for it in tqdm(range(num_iterations)):
            rng, rng_resample, rng_step_fn = jax.random.split(rng, 3)
            idx = int(jax.random.randint(
                    rng, shape=[], minval=0, maxval=self.dl.n_conditions
                ))
            batch = self.dl.sample_batch(idx, rng)
            src, tgt = batch["src_lin"], batch["tgt_lin"]
            src_cond = batch.get("src_condition")

            if self.model.match_fn is not None:
                tmat = self.model.match_fn(src, tgt)
                src_ixs, tgt_ixs = solver_utils.sample_joint(rng_resample, tmat)
                src, tgt = src[src_ixs], tgt[tgt_ixs]
                src_cond = None if src_cond is None else src_cond[src_ixs]

            self.model.vf_state, loss = self.model.step_fn(
                rng_step_fn,
                self.model.vf_state,
                src,
                tgt,
                src_cond,
            )

            training_logs["loss"].append(float(loss))
            if ((it-1) % valid_freq == 0) and (it > 1):
                train_loss = np.mean(training_logs["loss"][valid_freq :])
                log_metrics = {"train_loss": train_loss}
                
                callback_fn(
                    self.model,
                    log_metrics,
                )