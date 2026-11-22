"""
SimpleSLPClassifier - Single Layer Perceptron for Classification
"""

from typing import Optional, Tuple
import numpy as np
from numpy.typing import NDArray
from activations import logistic, softmax
from slp_base import BaseSLPEstimator


class SimpleSLPClassifier(BaseSLPEstimator):
    """
    Simple Single Layer Perceptron Classifier with one hidden layer.

    - 2 classes: one output neuron with a sigmoid, binary cross-entropy.
    - More classes: one output neuron per class with a softmax, categorical
      cross-entropy (kept because the provided test_multiclass requires it).

    Compatible interface with sklearn.neural_network.MLPClassifier.
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
        Initialize the SLP classifier.

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

        # Classifier-specific attributes
        self.classes_: Optional[NDArray[np.int_]] = None  # Unique class labels
        self.n_outputs_: Optional[int] = None  # Number of output neurons

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
        z2 = a1 @ self.W2_ + self.b2_  # output layer, shape (n_samples, n_outputs)
        y_pred = logistic(z2) if self.n_outputs_ == 1 else softmax(z2)
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
            One-hot encoded target (a 0/1 column in the binary case)
        z1, a1, z2, y_pred : arrays
            Values from forward propagation

        Returns:
        --------
        dW1, db1, dW2, db2 : tuple of arrays
            Gradients for weights and biases
        """
        _, f_derivative = self._get_activation_function()
        # Compute output layer error dL/dz2. With sigmoid + binary cross-entropy
        # (and softmax + categorical cross-entropy) it simplifies to y_pred - y.
        # Divided by n because the loss is a mean over the samples.
        delta2 = (y_pred - y) / len(X)
        # Compute hidden layer error dL/dz1: send delta2 back through W2, times f'(z1)
        delta1 = (delta2 @ self.W2_.T) * f_derivative(z1)
        # Compute gradients (the matrix products sum over the samples)
        dW2, db2 = a1.T @ delta2, delta2.sum(axis=0)
        dW1, db1 = X.T @ delta1, delta1.sum(axis=0)
        return dW1, db1, dW2, db2

    def _compute_loss(
        self, y_true: NDArray[np.floating], y_pred: NDArray[np.floating]
    ) -> float:
        """
        Compute cross-entropy loss.

        Parameters:
        -----------
        y_true : array-like, shape (n_samples, n_classes)
            One-hot encoded true labels
        y_pred : array-like, shape (n_samples, n_classes)
            Predicted probabilities

        Returns:
        --------
        loss : float
            Cross-entropy loss
        """
        # Clip predictions to avoid log(0)
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        if self.n_outputs_ == 1:
            # Binary: -[y log p + (1 - y) log(1 - p)]
            return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))
        # Multi-class: -sum_k y_k log p_k
        return float(-np.mean(np.sum(y_true * np.log(p), axis=1)))

    def fit(
        self, X: NDArray[np.floating], y: NDArray[np.int_]
    ) -> "SimpleSLPClassifier":
        """
        Fit the SLP classifier to training data.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target class labels

        Returns:
        --------
        self : object
            Fitted estimator
        """
        self._validate_params()
        X, y = self._check_X_y(X, y)

        # 1. Set random seed if provided (private generator => reproducible)
        self._rng = np.random.default_rng(self.random_state)

        # 2. Identify unique classes and encode y
        #    (y_index gives the class number 0..K-1 of each sample)
        self.classes_, y_index = np.unique(y, return_inverse=True)
        self.n_features_in_ = X.shape[1]
        self.n_outputs_ = 1 if len(self.classes_) <= 2 else len(self.classes_)
        if self.n_outputs_ == 1:
            Y = y_index.reshape(-1, 1).astype(float)  # a 0/1 column
        else:
            Y = np.eye(self.n_outputs_)[y_index]  # one-hot rows

        # 3. Initialize weights and 4. run the training loop
        #    (forward -> loss -> backward -> update, loss stored in loss_curve_)
        self._train(X, Y)
        return self

    def predict_proba(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Predict class probabilities for X.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples

        Returns:
        --------
        proba : array-like, shape (n_samples, n_classes)
            Class probabilities
        """
        # Use forward propagation and return the output probabilities
        y_pred = self._forward_propagation(self._check_X(X))[3]
        if self.n_outputs_ == 1:
            # The sigmoid gives P(class 1); P(class 0) = 1 - P(class 1)
            return np.hstack([1 - y_pred, y_pred])
        return y_pred

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.int_]:
        """
        Predict class labels for X.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples

        Returns:
        --------
        y_pred : array-like, shape (n_samples,)
            Predicted class labels
        """
        # Get probabilities and return class with highest probability (threshold 0.5 if binary)
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def score(self, X: NDArray[np.floating], y: NDArray[np.int_]) -> float:
        """
        Return the mean accuracy on the given test data and labels.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test samples
        y : array-like, shape (n_samples,)
            True labels

        Returns:
        --------
        score : float
            Mean accuracy
        """
        return float(np.mean(self.predict(X) == np.asarray(y)))
