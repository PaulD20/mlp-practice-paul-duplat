"""
Activation Functions and Their Derivatives
"""

import numpy as np
from numpy.typing import NDArray


def identity(z: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Identity (linear) activation function: f(z) = z.

    Parameters:
    -----------
    z : array-like
        Input values

    Returns:
    --------
    array-like
        Activated values (unchanged)
    """
    return z


def identity_derivative(z: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Derivative of identity activation: 1 everywhere.

    Parameters:
    -----------
    z : array-like
        Input values (pre-activation)

    Returns:
    --------
    array-like
        Gradient values
    """
    return np.ones_like(z)


def relu(z: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    ReLU activation function.

    Parameters:
    -----------
    z : array-like
        Input values

    Returns:
    --------
    array-like
        Activated values
    """
    return np.maximum(0.0, z)


def relu_derivative(z: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Derivative of ReLU activation.

    Parameters:
    -----------
    z : array-like
        Input values (pre-activation)

    Returns:
    --------
    array-like
        Gradient values
    """
    # 1 if z > 0, else 0 (convention: 0 at z = 0, where ReLU is not differentiable)
    return (z > 0).astype(float)


def tanh(z: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Hyperbolic tangent activation function.

    Parameters:
    -----------
    z : array-like
        Input values

    Returns:
    --------
    array-like
        Activated values
    """
    return np.tanh(z)


def tanh_derivative(z: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Derivative of tanh activation.

    Parameters:
    -----------
    z : array-like
        Input values (pre-activation)

    Returns:
    --------
    array-like
        Gradient values
    """
    return 1.0 - np.tanh(z) ** 2


def logistic(z: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Logistic (sigmoid) activation function.

    Parameters:
    -----------
    z : array-like
        Input values

    Returns:
    --------
    array-like
        Activated values
    """
    # Use np.clip to avoid overflow in exp(-z)
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def logistic_derivative(z: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Derivative of logistic activation.

    Parameters:
    -----------
    z : array-like
        Input values (pre-activation)

    Returns:
    --------
    array-like
        Gradient values
    """
    s = logistic(z)
    return s * (1.0 - s)


def softmax(z: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Softmax activation for output layer (classification).

    Parameters:
    -----------
    z : array-like, shape (n_samples, n_classes)
        Input values

    Returns:
    --------
    array-like, shape (n_samples, n_classes)
        Probabilities that sum to 1 for each sample
    """
    # Subtracting the row maximum does not change the result but prevents exp() from overflowing
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)
