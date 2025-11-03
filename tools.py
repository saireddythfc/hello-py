# tools.py
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def get_data(seed=42):
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        random_state=seed,
    )
    return train_test_split(X, y, test_size=0.2, random_state=seed)

X_train, X_val, y_train, y_val = get_data()

def train_and_eval(params):
    """Train logistic regression with given params; return val accuracy."""
    lr = params.get("lr", 0.01)
    wd = params.get("weight_decay", 0.01)

    model = LogisticRegression(
        solver="lbfgs",
        C=1.0 / (wd + 1e-5),
        max_iter=200,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return accuracy_score(y_val, preds)
