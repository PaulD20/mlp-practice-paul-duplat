"""
SimpleSLPRegressor - Single Layer Perceptron for Regression
"""

from typing import Optional, Tuple
import numpy as np
from numpy.typing import NDArray
from slp_base import BaseSLPEstimator


class SimpleSLPRegressor(BaseSLPEstimator):
    """
    Simple Single Layer Perceptron Regressor with one hidden layer.

    Output layer: one neuron with the identity activation. Loss: mean squared
    error. The target is standardized inside fit and converted back in
    predict, so the user always works in the original units.

    Compatible interface with sklearn.neural_network.MLPRegressor.
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
        Initialize the SLP regressor.

        Parameters:
        -----------
        hidden_layer_size : int
            Number of neurons in the hidden layer
        activation : str
            Activation function ('identity', 'logistic', 'tanh', 'relu'}, default='logistic')
        learning_rate : float
            Learning rate for gradient descent
        max_iter : int
            Maximum number of iterations
        random_state : int or None
            Random seed for reproducibility
        batch_size, early_stopping, validation_fraction, n_iter_no_change, tol
            See BaseSLPEstimator
        """
        super().__init__(
            hidden_layer_size, activation, learning_rate, max_iter, random_state,
            batch_size, early_stopping, validation_fraction, n_iter_no_change, tol,
        )

    def _forward_propagation(self, X: NDArray[np.floating]) -> Tuple[
        NDArray[np.floating],
        NDArray[np.floating],
        NDArray[np.floating],
        NDArray[np.floating],
    ]:
        """
        Perform forward propagation.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data

        Returns:
        --------
        z1, a1, z2, y_pred : tuple of arrays
            Intermediate values for backpropagation
        """
        f, _ = self._get_activation_function()
        z1 = X @ self.W1_ + self.b1_  # hidden layer, shape (n_samples, hidden_layer_size)
        a1 = f(z1)
        z2 = a1 @ self.W2_ + self.b2_  # output layer, shape (n_samples, 1)
        y_pred = z2  # identity output: a regression output is not bounded
        return z1, a1, z2, y_pred

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

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data
        y : array-like, shape (n_samples, n_outputs)
            Target values
        z1, a1, z2, y_pred : arrays
            Values from forward propagation

        Returns:
        --------
        dW1, db1, dW2, db2 : tuple of arrays
            Gradients for weights and biases
        """
        _, f_derivative = self._get_activation_function()
        # Output layer error dL/dz2. Loss = 1/(2n) * sum (y_pred - y)^2 and
        # the identity has derivative 1, so dL/dz2 = (y_pred - y) / n.
        delta2 = (y_pred - y) / len(X)
        # Hidden layer error dL/dz1: send delta2 back through W2, times f'(z1)
        delta1 = (delta2 @ self.W2_.T) * f_derivative(z1)
        # Gradients (the matrix products sum over the samples)
        dW2, db2 = a1.T @ delta2, delta2.sum(axis=0)
        dW1, db1 = X.T @ delta1, delta1.sum(axis=0)
        return dW1, db1, dW2, db2

    def _compute_loss(
        self, y_true: NDArray[np.floating], y_pred: NDArray[np.floating]
    ) -> float:
        """
        Compute mean squared error loss.

        Uses 1/2 * MSE like scikit-learn: the 1/2 cancels the 2 that appears
        when differentiating and does not move the minimum.

        Parameters:
        -----------
        y_true : array-like
            True values
        y_pred : array-like
            Predicted values

        Returns:
        --------
        loss : float
            MSE loss
        """
        return float(0.5 * np.mean((y_true - y_pred) ** 2))

    def fit(
        self, X: NDArray[np.floating], y: NDArray[np.floating]
    ) -> "SimpleSLPRegressor":
        """
        Fit the SLP regressor to training data.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values

        Returns:
        --------
        self : object
            Fitted estimator
        """
        self._validate_params()
        X, y = self._check_X_y(X, y)
        y = y.astype(float)

        # Set random seed if provided (private generator => reproducible)
        self._rng = np.random.default_rng(self.random_state)
        self.n_features_in_ = X.shape[1]
        self.n_outputs_ = 1

        # Standardize the target (tip of the brief): the network learns a
        # variable with mean 0 and std 1, whatever the original scale.
        self.y_mean_ = y.mean()
        self.y_std_ = y.std() if y.std() > 0 else 1.0  # constant target: avoid / 0
        Y = ((y - self.y_mean_) / self.y_std_).reshape(-1, 1)

        # Initialize weights and run the training loop (similar to classifier)
        self._train(X, Y)
        return self

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Predict using the trained model.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples

        Returns:
        --------
        y_pred : array-like, shape (n_samples,)
            Predicted values
        """
        y_scaled = self._forward_propagation(self._check_X(X))[3]
        return y_scaled.ravel() * self.y_std_ + self.y_mean_  # back to the original units

    def score(self, X: NDArray[np.floating], y: NDArray[np.floating]) -> float:
        """
        Return the R² score on the given test data.

        R² = 1 - sum (y - y_pred)^2 / sum (y - mean(y))^2

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test samples
        y : array-like, shape (n_samples,)
            True values

        Returns:
        --------
        score : float
            R² score
        """
        y = np.asarray(y, dtype=float)
        ss_res = np.sum((y - self.predict(X)) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return float(1 - ss_res / ss_tot)
