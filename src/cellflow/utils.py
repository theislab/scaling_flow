from typing import Any, Literal

import jax
import jax.numpy as jnp
from ott.geometry import costs, pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
from flax.core.frozen_dict import freeze, unfreeze

ScaleCost_t = float | Literal["mean", "max_cost", "median"]

__all__ = ["match_linear", "default_prng_key"]


def match_linear(
    source_batch: jnp.ndarray,
    target_batch: jnp.ndarray,
    cost_fn: costs.CostFn | None = costs.SqEuclidean(),
    epsilon: float | None = 1.0,
    scale_cost: ScaleCost_t = "mean",
    tau_a: float = 1.0,
    tau_b: float = 1.0,
    threshold: float | None = None,
    **kwargs: Any,
) -> jnp.ndarray:
    """Compute solution to a linear OT problem.

    Parameters
    ----------
    source_batch
        Source point cloud of shape ``[n, d]``.
    target_batch
        Target point cloud of shape ``[m, d]``.
    cost_fn
        Cost function to use for the linear OT problem.
    epsilon
        Regularization parameter.
    scale_cost
        Scaling of the cost matrix.
    tau_a
        Parameter in :math:`(0, 1]` that defines how unbalanced the problem is
        in the source distribution. If :math:`1`, the problem is balanced in the source distribution.
    tau_b
        Parameter in :math:`(0, 1]` that defines how unbalanced the problem is in the target
        distribution. If :math:`1`, the problem is balanced in the target distribution.
    threshold
        Convergence criterion for the Sinkhorn algorithm.
    kwargs
        Additional arguments for :class:`ott.solvers.linear.sinkhorn.Sinkhorn`.

    Returns
    -------
    Optimal transport matrix between ``'source_batch'`` and ``'target_batch'``.
    """
    if threshold is None:
        threshold = 1e-3 if (tau_a == 1.0 and tau_b == 1.0) else 1e-2
    geom = pointcloud.PointCloud(
        source_batch,
        target_batch,
        cost_fn=cost_fn,
        epsilon=epsilon,
        scale_cost=scale_cost,
    )
    problem = linear_problem.LinearProblem(geom, tau_a=tau_a, tau_b=tau_b)
    solver = sinkhorn.Sinkhorn(threshold=threshold, **kwargs)
    out = solver(problem)
    return out.matrix


def default_prng_key(rng: jax.Array | None) -> jax.Array:
    """Get the default PRNG key.

    Parameters
    ----------
    rng: PRNG key.

    Returns
    -------
      If ``rng = None``, returns the default PRNG key. Otherwise, it returns
      the unmodified ``rng`` key.
    """
    return jax.random.key(0) if rng is None else rng



def convert_prophet_weights_to_flax(
    prophet_checkpoint_path: str,
    flax_params: Dict,
    model_config: Dict[str, Any]
) -> Dict:
    """
    Convert Prophet PyTorch weights to JAX/Flax format, placing parameters
    within the 'condition_encoder' category.

    Parameters
    ----------
    prophet_checkpoint_path : str
        Path to Prophet checkpoint.
    flax_params : Dict
        Structure of Flax parameters to fill.
    model_config : Dict[str, Any]
        Configuration for model architecture (to align parameters).

    Returns
    -------
    Dict
        Flax parameters with weights from Prophet.
    """
    # Check if the checkpoint exists
    if not os.path.exists(prophet_checkpoint_path):
        raise FileNotFoundError(f"Prophet checkpoint not found at {prophet_checkpoint_path}")

    # Load PyTorch checkpoint
    try:
        checkpoint = torch.load(prophet_checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    except Exception as e:
        raise ValueError(f"Error loading PyTorch checkpoint: {e}")

    # Create a mutable copy of flax_params
    new_params = unfreeze(flax_params)

    # Make sure we have a condition_encoder key
    if "condition_encoder" not in new_params:
        new_params["condition_encoder"] = {}

    for tokenizer_name in model_config.get("tokenizer_config", {}).keys():
        print(f"\nProcessing tokenizer: {tokenizer_name}")
        # Handle each tokenizer network
        torch_prefix = f"tokenizer_nets.{tokenizer_name}"
        flax_prefix = f"tokenizer_nets_{tokenizer_name}"

        # Check for different weight key patterns - either direct or with mlp
        mlp_keys = [k for k in state_dict.keys() if k.startswith(f"{torch_prefix}.mlp")]
        direct_keys = [k for k in state_dict.keys() if k.startswith(torch_prefix) and not ".mlp" in k]

        # Choose the pattern that has keys
        if len(mlp_keys) > 0:
            key_pattern = "mlp"
        elif len(direct_keys) > 0:
            key_pattern = "direct"
        else:
            print(f"No weights found for {tokenizer_name} tokenizer")
            continue

        # Check if the Flax structure already exists and get layer naming pattern
        has_flax_structure = flax_prefix in new_params["condition_encoder"]
        expected_layer_names = []
        use_dense_pattern = False

        if has_flax_structure:
            expected_layer_names = list(new_params["condition_encoder"][flax_prefix].keys())
            # Check if layer names follow Dense_X pattern or layers_X pattern
            if any(name.startswith("Dense_") for name in expected_layer_names):
                use_dense_pattern = True

        processed_layers = []

        # Process the weights according to the pattern
        if key_pattern == "mlp":
            layer_idx = 0
            while True:
                # Linear layers in PyTorch MLP are every 3rd layer (Linear->GELU->Dropout pattern)
                weight_key = f"{torch_prefix}.mlp.{layer_idx*3}.weight"
                bias_key = f"{torch_prefix}.mlp.{layer_idx*3}.bias"

                if weight_key not in state_dict or bias_key not in state_dict:
                    break

                # Convert weights to JAX arrays
                weight = state_dict[weight_key].detach().numpy()
                bias = state_dict[bias_key].detach().numpy()

                # Determine layer name based on pattern in the Flax model
                if use_dense_pattern:
                    flax_layer_name = f"Dense_{layer_idx}"
                    # For Dense_X, STILL TRANSPOSE the PyTorch weights (which are out_features, in_features)
                    transposed_weight = weight.T
                else:
                    flax_layer_name = f"layers_{layer_idx}"
                    # For layers_X, transpose to JAX format (in_features, out_features)
                    transposed_weight = weight.T

                # Update parameters
                if flax_prefix not in new_params["condition_encoder"]:
                    new_params["condition_encoder"][flax_prefix] = {}

                if flax_layer_name not in new_params["condition_encoder"][flax_prefix]:
                    new_params["condition_encoder"][flax_prefix][flax_layer_name] = {}

                new_params["condition_encoder"][flax_prefix][flax_layer_name]["kernel"] = jnp.array(transposed_weight)
                new_params["condition_encoder"][flax_prefix][flax_layer_name]["bias"] = jnp.array(bias)

                processed_layers.append(flax_layer_name)

                layer_idx += 1

        # If we found existing structure but didn't process all layers, add a warning
        if has_flax_structure and set(processed_layers) != set(expected_layer_names):
            missing_layers = set(expected_layer_names) - set(processed_layers)
            print(f"  Warning: Did not process all expected layers for {tokenizer_name}. Missing: {missing_layers}")

        # For verification purposes, add a check of how this matches with original params
        weight_key = f"{torch_prefix}.mlp.0.weight" if key_pattern == "mlp" else f"{torch_prefix}.0.weight"
        if weight_key in state_dict:
            weight = state_dict[weight_key].detach().numpy()

    # Map learnable embeddings - within condition_encoder
    for col in model_config.get("learnable_columns", []):
        torch_key = f"learnable_embeddings.{col}.weight"
        if torch_key in state_dict:
            emb_weight = jnp.array(state_dict[torch_key].detach().numpy())

            flax_key = f"learnable_embeddings_{col}"
            if flax_key not in new_params["condition_encoder"]:
                new_params["condition_encoder"][flax_key] = {}

            new_params["condition_encoder"][flax_key]["embedding"] = emb_weight

    # Map CLS token - within condition_encoder
    if "cls_token" in state_dict:
        cls_weight = jnp.array(state_dict["cls_token"].detach().squeeze().numpy())
        new_params["condition_encoder"]["cls_token"] = cls_weight

    # Map transformer encoder layers - within condition_encoder
    if "transformer" not in new_params["condition_encoder"]:
        # Create the structure that matches the ProphetEncoder class
        # TransformerEncoder(nn.Module) doesn't have an "encoderblock" key
        new_params["condition_encoder"]["transformer"] = {}

    for layer_idx in range(model_config.get("num_layers", 0)):
        # In PyTorch: transformer.layers.{layer_idx}.*
        # Map self-attention
        torch_q_key = f"transformer.layers.{layer_idx}.self_attn.in_proj_weight"
        torch_q_bias = f"transformer.layers.{layer_idx}.self_attn.in_proj_bias"
        torch_out_key = f"transformer.layers.{layer_idx}.self_attn.out_proj.weight"
        torch_out_bias = f"transformer.layers.{layer_idx}.self_attn.out_proj.bias"

        if torch_q_key in state_dict and torch_out_key in state_dict:
            # PyTorch packs q,k,v into one tensor - split them
            qkv_weight = state_dict[torch_q_key].detach().numpy()
            qkv_bias = state_dict[torch_q_bias].detach().numpy()

            # Split into q, k, v
            model_dim = model_config.get("model_dim", 256)
            q_weight = jnp.array(qkv_weight[:model_dim]).T
            k_weight = jnp.array(qkv_weight[model_dim:2*model_dim]).T
            v_weight = jnp.array(qkv_weight[2*model_dim:]).T

            q_bias = jnp.array(qkv_bias[:model_dim])
            k_bias = jnp.array(qkv_bias[model_dim:2*model_dim])
            v_bias = jnp.array(qkv_bias[2*model_dim:])

            # Output projection
            out_weight = jnp.array(state_dict[torch_out_key].detach().numpy()).T
            out_bias = jnp.array(state_dict[torch_out_bias].detach().numpy())

            # Update parameters within condition_encoder/transformer
            # The structure should match what ProphetEncoder expects
            if "layers" not in new_params["condition_encoder"]["transformer"]:
                new_params["condition_encoder"]["transformer"]["layers"] = {}

            if str(layer_idx) not in new_params["condition_encoder"]["transformer"]["layers"]:
                new_params["condition_encoder"]["transformer"]["layers"][str(layer_idx)] = {}

            layer_params = new_params["condition_encoder"]["transformer"]["layers"][str(layer_idx)]

            # Create the attention structure
            if "self_attn" not in layer_params:
                layer_params["self_attn"] = {}

            layer_params["self_attn"]["query"] = {"kernel": q_weight, "bias": q_bias}
            layer_params["self_attn"]["key"] = {"kernel": k_weight, "bias": k_bias}
            layer_params["self_attn"]["value"] = {"kernel": v_weight, "bias": v_bias}
            layer_params["self_attn"]["out_proj"] = {"kernel": out_weight, "bias": out_bias}

        # Map MLP layers
        torch_fc1_key = f"transformer.layers.{layer_idx}.linear1.weight"
        torch_fc1_bias = f"transformer.layers.{layer_idx}.linear1.bias"
        torch_fc2_key = f"transformer.layers.{layer_idx}.linear2.weight"
        torch_fc2_bias = f"transformer.layers.{layer_idx}.linear2.bias"

        if torch_fc1_key in state_dict and torch_fc2_key in state_dict:
            fc1_weight = jnp.array(state_dict[torch_fc1_key].detach().numpy()).T
            fc1_bias = jnp.array(state_dict[torch_fc1_bias].detach().numpy())
            fc2_weight = jnp.array(state_dict[torch_fc2_key].detach().numpy()).T
            fc2_bias = jnp.array(state_dict[torch_fc2_bias].detach().numpy())

            # If the layers dict doesn't exist yet, create it
            if "layers" not in new_params["condition_encoder"]["transformer"]:
                new_params["condition_encoder"]["transformer"]["layers"] = {}

            if str(layer_idx) not in new_params["condition_encoder"]["transformer"]["layers"]:
                new_params["condition_encoder"]["transformer"]["layers"][str(layer_idx)] = {}

            layer_params = new_params["condition_encoder"]["transformer"]["layers"][str(layer_idx)]

            # Create the MLP structure
            if "mlp" not in layer_params:
                layer_params["mlp"] = {}

            layer_params["mlp"]["c_fc"] = {"kernel": fc1_weight, "bias": fc1_bias}
            layer_params["mlp"]["c_proj"] = {"kernel": fc2_weight, "bias": fc2_bias}

        # Map layer norms
        torch_ln1_key = f"transformer.layers.{layer_idx}.norm1.weight"
        torch_ln1_bias = f"transformer.layers.{layer_idx}.norm1.bias"
        torch_ln2_key = f"transformer.layers.{layer_idx}.norm2.weight"
        torch_ln2_bias = f"transformer.layers.{layer_idx}.norm2.bias"

        if torch_ln1_key in state_dict and torch_ln2_key in state_dict:
            ln1_weight = jnp.array(state_dict[torch_ln1_key].detach().numpy())
            ln1_bias = jnp.array(state_dict[torch_ln1_bias].detach().numpy())
            ln2_weight = jnp.array(state_dict[torch_ln2_key].detach().numpy())
            ln2_bias = jnp.array(state_dict[torch_ln2_bias].detach().numpy())

            # If the layers dict doesn't exist yet, create it
            if "layers" not in new_params["condition_encoder"]["transformer"]:
                new_params["condition_encoder"]["transformer"]["layers"] = {}

            if str(layer_idx) not in new_params["condition_encoder"]["transformer"]["layers"]:
                new_params["condition_encoder"]["transformer"]["layers"][str(layer_idx)] = {}

            layer_params = new_params["condition_encoder"]["transformer"]["layers"][str(layer_idx)]

            # Add the layer norms
            layer_params["norm1"] = {"scale": ln1_weight, "bias": ln1_bias}
            layer_params["norm2"] = {"scale": ln2_weight, "bias": ln2_bias}

    return freeze(new_params)
