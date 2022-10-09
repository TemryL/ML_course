# -*- coding: utf-8 -*-
"""a function used to compute the loss."""

import numpy as np


def calculate_mse(e):
    """Calculate the mse for vector e."""
    return np.mean(e.dot(e)) / 2


def calculate_mae(e):
    """Calculate the mae for vector e."""
    return np.mean(np.abs(e))


def compute_loss(y, tx, w, type="mse"):
    """Calculate the loss using either MSE or MAE.
    
    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.
        type: string in ["mae", "mse"] specifying the type of loss to compute
    
    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    
    e = y - tx @ w
    
    if type == "mse":
        return calculate_mse(e)
    
    elif type == "mae":
        return calculate_mae(e)
    
    else:
        raise ValueError("Invalid value for argument 'type' when calling compute_loss, 'type' must be in ['mse', 'mae'].")