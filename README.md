# One-Hidden-Layer Neural Network from Scratch

NumPy implementation of a neural network with **one hidden layer** for binary classification and
regression, with an interface close to scikit-learn's `MLPClassifier` / `MLPRegressor`
(`fit` / `predict` / `predict_proba` / `score`, learned attributes ending in `_`, `random_state`).

Training: forward propagation, backpropagation (chain rule), mini-batch stochastic gradient descent,
optional early stopping. Losses: cross-entropy (classification), mean squared error (regression).

## Project structure

| File | Content |
|---|---|
| `activations.py` | `logistic`, `relu`, `tanh`, `identity`, `softmax` and their derivatives (vectorized) |
| `slp_base.py` | `BaseSLPEstimator`: hyper-parameters and their validation, Glorot initialization, mini-batch SGD loop with early stopping |
| `slp_classifier.py` | `SimpleSLPClassifier`: forward/backward pass, sigmoid output + binary cross-entropy (softmax for > 2 classes) |
| `slp_regressor.py` | `SimpleSLPRegressor`: forward/backward pass, identity output + MSE, target standardized internally |
| `test_*.py` | Provided tests (pytest) |
| `report.ipynb` | Report: debugging tests, diabetes, breast cancer, two Kaggle datasets, comparison with scikit-learn, early stopping, batch size |
| `data/` | Kaggle datasets: `UCI_Credit_Card.csv` (classification), `housing.csv` (regression) |

The model files use NumPy only; scikit-learn, pandas and matplotlib are used in the tests and the report.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```python
from slp_classifier import SimpleSLPClassifier

clf = SimpleSLPClassifier(hidden_layer_size=32, activation="relu", learning_rate=0.01,
                          batch_size=32, max_iter=100, random_state=42)
clf.fit(X_train, y_train)          # returns self
clf.predict_proba(X_test)          # (n_samples, 2)
clf.score(X_test, y_test)          # accuracy
clf.loss_curve_                    # training loss per epoch
```

## Run the tests

```bash
pytest -q
```

## Run the report

```bash
MPLBACKEND=Agg ipython report.ipynb                                    # print all results in the terminal (no plots)
ipython report.ipynb                                                   # same, plots open in windows (close each to continue)
jupyter nbconvert --to notebook --execute report.ipynb --inplace       # save results and plots inside the notebook
```
