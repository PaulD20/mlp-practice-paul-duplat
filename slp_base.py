"""
Base class for SLP Classifier and Regressor
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple
import numpy as np
from numpy.typing import NDArray
from activations import (
    identity,
    identity_derivative,
    relu,
    relu_derivative,
    tanh,
    tanh_derivative,
    logistic,
    logistic_derivative,
)

VALID_ACTIVATIONS = ("identity", "logistic", "tanh", "relu")


class BaseSLPEstimator(ABC):
    """
    Abstract base class for SLP Classifier and Regressor.

    Contains common functionality shared between both estimators:
    hyper-parameters, checks, weight initialization and the training loop
    (mini-batch stochastic gradient descent with optional early stopping).
    """

    def __init__(
        self,
        hidden_layer_size: int = 100,
        activation: str = "logistic",
        learning_rate: float = 0.05,
        max_iter: int = 200,
        random_state: Optional[int] = None,
        batch_size: int = 32,
        early_stopping: bool = False,
        validation_fraction: float = 0.1,
        n_iter_no_change: int = 10,
        tol: float = 1e-4,
    ) -> None:
        """
        Initialize the SLP estimator.

        Like scikit-learn, hyper-parameters are stored as given and only
        checked when fit() is called.

        Parameters:
        -----------
        hidden_layer_size : int
            Number of neurons in the hidden layer
        activation : str
            Activation function ('identity', 'logistic', 'tanh', 'relu'}, default='logistic')
        learning_rate : float
            Learning rate for gradient descent
        max_iter : int
            Maximum number of iterations (epochs = full passes over the training data)
        random_state : int or None
            Random seed for reproducibility
        batch_size : int
            Samples per gradient step; batch_size >= n_samples gives full-batch descent
        early_stopping : bool
            If True, stop when the loss on a held-out validation set stops improving
        validation_fraction : float in (0, 1)
            Share of the training data held out for early stopping
        n_iter_no_change : int
            Number of epochs without improvement before stopping
        tol : float
            Minimum decrease of the validation loss that counts as an improvement
        """
        self.hidden_layer_size: int = hidden_layer_size
        self.activation: str = activation
        self.learning_rate: float = learning_rate
        self.max_iter: int = max_iter
        self.random_state: Optional[int] = random_state
        self.batch_size: int = batch_size
        self.early_stopping: bool = early_stopping
        self.validation_fraction: float = validation_fraction
        self.n_iter_no_change: int = n_iter_no_change
        self.tol: float = tol

        # To be initialized in fit()
        self.W1_: Optional[NDArray[np.floating]] = None  # Weights: input -> hidden
        self.b1_: Optional[NDArray[np.floating]] = None  # Biases: hidden layer
        self.W2_: Optional[NDArray[np.floating]] = None  # Weights: hidden -> output
        self.b2_: Optional[NDArray[np.floating]] = None  # Biases: output layer
        self.loss_curve_: list[float] = []  # Track loss during training

    def _get_activation_function(
        self,
    ) -> Tuple[Callable[[NDArray], NDArray], Callable[[NDArray], NDArray]]:
        """Return the activation function and its derivative."""
        if self.activation == "relu":
            return relu, relu_derivative
        elif self.activation == "tanh":
            return tanh, tanh_derivative
        elif self.activation == "logistic":
            return logistic, logistic_derivative
        elif self.activation == "identity":
            return identity, identity_derivative
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def _initialize_weights(self) -> None:
        """
        Initialize weights and biases.

        Glorot (Xavier) uniform initialization, as in scikit-learn:
        weights ~ U(-limit, limit) with limit = sqrt(6 / (fan_in + fan_out)),
        so the scale of the signal stays about the same from one layer to
        the next. Random weights break the symmetry between hidden neurons;
        biases start at 0.

        (A fixed N(0, 0.01^2) initialization was tried first: with plain SGD
        the network stayed on a near-linear plateau and several tests failed.)
        """
        limit1 = np.sqrt(6 / (self.n_features_in_ + self.hidden_layer_size))
        limit2 = np.sqrt(6 / (self.hidden_layer_size + self.n_outputs_))
        self.W1_ = self._rng.uniform(-limit1, limit1, size=(self.n_features_in_, self.hidden_layer_size))
        self.b1_ = np.zeros(self.hidden_layer_size)
        self.W2_ = self._rng.uniform(-limit2, limit2, size=(self.hidden_layer_size, self.n_outputs_))
        self.b2_ = np.zeros(self.n_outputs_)

    # ------------------------------------------------------------------
    # Checks
    # ------------------------------------------------------------------

    def _validate_params(self) -> None:
        """Raise a ValueError if a hyper-parameter is invalid."""
        if not (isinstance(self.hidden_layer_size, int) and self.hidden_layer_size >= 1):
            raise ValueError(f"hidden_layer_size must be a positive integer, got {self.hidden_layer_size!r}")
        if self.activation not in VALID_ACTIVATIONS:
            raise ValueError(f"activation must be one of {VALID_ACTIVATIONS}, got {self.activation!r}")
        if not self.learning_rate > 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate!r}")
        if not (isinstance(self.max_iter, int) and self.max_iter >= 1):
            raise ValueError(f"max_iter must be a positive integer, got {self.max_iter!r}")
        if not (isinstance(self.batch_size, int) and self.batch_size >= 1):
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size!r}")
        if not 0 < self.validation_fraction < 1:
            raise ValueError(f"validation_fraction must be in (0, 1), got {self.validation_fraction!r}")
        if not (isinstance(self.n_iter_no_change, int) and self.n_iter_no_change >= 1):
            raise ValueError(f"n_iter_no_change must be a positive integer, got {self.n_iter_no_change!r}")
        if not self.tol >= 0:
            raise ValueError(f"tol must be >= 0, got {self.tol!r}")

    @staticmethod
    def _check_X_y(X: NDArray, y: NDArray) -> Tuple[NDArray[np.floating], NDArray]:
        """Convert the training data to arrays and check their shapes (used by fit)."""
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
            raise ValueError(f"expected X of shape (n_samples, n_features) and y of shape (n_samples,), "
                             f"got {X.shape} and {y.shape}")
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains NaN or infinite values")
        return X, y

    def _check_X(self, X: NDArray) -> NDArray[np.floating]:
        """Convert X to an array and check the model is fitted with the same number of features."""
        if self.W1_ is None:
            raise ValueError(f"This {type(self).__name__} is not fitted yet. Call fit() first.")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(f"expected X of shape (n_samples, {self.n_features_in_}), got {X.shape}")
        return X

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _train(self, X: NDArray[np.floating], Y: NDArray[np.floating]) -> None:
        """
        Training loop shared by both models (mini-batch stochastic gradient descent).

        X : array, shape (n_samples, n_features)
        Y : array, shape (n_samples, n_outputs), targets already encoded

        Each epoch: shuffle the data, cut it into mini-batches and, for each
        batch, run forward -> backward -> update. The loss on the whole
        training set is recorded after each epoch in loss_curve_.
        """
        self._initialize_weights()
        self.loss_curve_ = []
        self.validation_loss_curve_: list[float] = []

        if self.early_stopping:
            # Hold out part of the training data to measure overfitting
            n_val = max(1, int(round(self.validation_fraction * len(X))))
            if n_val >= len(X):
                raise ValueError("not enough samples to hold out a validation set")
            perm = self._rng.permutation(len(X))
            X_val, Y_val = X[perm[:n_val]], Y[perm[:n_val]]
            X, Y = X[perm[n_val:]], Y[perm[n_val:]]
            best_val_loss, epochs_no_improve = np.inf, 0

        for epoch in range(self.max_iter):
            order = self._rng.permutation(len(X))  # new mini-batches at every epoch
            for start in range(0, len(X), self.batch_size):
                batch = order[start:start + self.batch_size]
                # Forward propagation
                z1, a1, z2, y_pred = self._forward_propagation(X[batch])
                # Backward propagation
                gradients = self._backward_propagation(X[batch], Y[batch], z1, a1, z2, y_pred)
                # Update weights: parameter <- parameter - learning_rate * gradient
                for param, grad in zip((self.W1_, self.b1_, self.W2_, self.b2_), gradients):
                    param -= self.learning_rate * grad

            # Compute loss and store it in loss_curve_
            self.loss_curve_.append(self._compute_loss(Y, self._forward_propagation(X)[3]))
            self.n_iter_ = epoch + 1

            if self.early_stopping:
                val_loss = self._compute_loss(Y_val, self._forward_propagation(X_val)[3])
                self.validation_loss_curve_.append(val_loss)
                if val_loss < best_val_loss - self.tol:
                    best_val_loss, epochs_no_improve = val_loss, 0
                    best_weights = [p.copy() for p in (self.W1_, self.b1_, self.W2_, self.b2_)]
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.n_iter_no_change:
                        break

        if self.early_stopping:
            # Keep the weights of the best validation epoch, not the last one
            self.W1_, self.b1_, self.W2_, self.b2_ = best_weights

    # ------------------------------------------------------------------
    # Implemented by subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def _forward_propagation(self, X: NDArray[np.floating]) -> Tuple[
        NDArray[np.floating],
        NDArray[np.floating],
        NDArray[np.floating],
        NDArray[np.floating],
    ]:
        """
        Perform forward propagation.

        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def _backward_propagation(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
        z1: NDArray[np.floating],
        a1: NDArray[np.floating],
        z2: NDArray[np.floating],
        y_pred: NDArray[np.floating],
    ) -> Tuple[
        NDArray[np.floating],
        NDArray[np.floating],
        NDArray[np.floating],
        NDArray[np.floating],
    ]:
        """
        Perform backpropagation to compute gradients.

        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def _compute_loss(
        self, y_true: NDArray[np.floating], y_pred: NDArray[np.floating]
    ) -> float:
        """
        Compute loss.

        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def fit(self, X: NDArray[np.floating], y: NDArray) -> "BaseSLPEstimator":
        """
        Fit the model to training data.

        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def predict(self, X: NDArray[np.floating]) -> NDArray:
        """
        Make predictions.

        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def score(self, X: NDArray[np.floating], y: NDArray) -> float:
        """
        Score the model.

        Must be implemented by subclasses.
        """
        pass
