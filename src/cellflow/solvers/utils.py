import jax


def ema_update(current_model, new_model, ema):
    """
    Update parameters using exponential moving average.

    Parameters
    ----------
        current_model
            Current parameters.
        new_model
            New parameters to be averaged.
        ema
            Exponential moving average factor
            between `0` and `1`. `0` means no update, `1` means full update.

    Returns
    -------
        Updated parameters after applying EMA.
    """
    new_target_params = jax.tree_map(lambda p, tp: p * (1 - ema) + tp * ema, current_model.params, new_model.params)
    return new_model.replace(params=new_target_params)
