import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from poliflow.core.Tensor import Tensor
from poliflow.nn.Activacion import ReLU, Sigmoide
from poliflow.nn.Lineal import Lineal
from poliflow.nn.Secuencial import Secuencial
from poliflow.optim.SGD import SGD
from poliflow.perdidas.BCE import BCE


def stratified_split(x, y, test_size=0.2, seed=123):
    rng = np.random.default_rng(seed)
    train_idx = []
    test_idx = []

    for klass in np.unique(y):
        idx = np.where(y.ravel() == klass)[0]
        rng.shuffle(idx)
        n_test = int(round(len(idx) * test_size))
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])

    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]


def binary_cross_entropy_np(y_pred, y_true):
    eps = 1e-7
    y_pred = np.clip(y_pred, eps, 1.0 - eps)
    return float(-(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)).mean())


def classification_metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    y_true = y_true.astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    np.random.seed(0)

    data_path = ROOT / "tests" / "data" / "diabetes.csv"
    df = pd.read_csv(data_path)

    x = df.drop(columns=["Outcome"]).to_numpy(dtype=float)
    y = df[["Outcome"]].to_numpy(dtype=float)

    x_train, x_test, y_train, y_test = stratified_split(x, y, test_size=0.2, seed=123)

    x_mean = x_train.mean(axis=0, keepdims=True)
    x_std = x_train.std(axis=0, keepdims=True)
    x_std[x_std == 0] = 1.0
    x_train = (x_train - x_mean) / x_std
    x_test = (x_test - x_mean) / x_std

    model = Secuencial(
        Lineal(8, 16),
        ReLU(),
        Lineal(16, 8),
        ReLU(),
        Lineal(8, 1),
        Sigmoide(),
    )
    criterion = BCE()
    optimizer = SGD(model.parameters(), lr=0.05)

    x_train_t = Tensor(x_train)
    y_train_t = Tensor(y_train)

    epochs = 500
    start = time.perf_counter()
    for _ in range(epochs):
        y_pred = model(x_train_t)
        loss = criterion(y_pred, y_train_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    elapsed = time.perf_counter() - start

    train_prob = model(Tensor(x_train)).data
    test_prob = model(Tensor(x_test)).data

    train_loss = binary_cross_entropy_np(train_prob, y_train)
    test_loss = binary_cross_entropy_np(test_prob, y_test)
    metrics = classification_metrics(y_test, test_prob)

    print("PoliFlow - Clasificacion binaria diabetes")
    print(f"Train Loss: {train_loss:.6f}")
    print(f"Test Loss:  {test_loss:.6f}")
    print(f"Tiempo (s): {elapsed:.4f}")
    print(f"Accuracy:   {metrics['accuracy']:.6f}")
    print(f"Precision:  {metrics['precision']:.6f}")
    print(f"Recall:     {metrics['recall']:.6f}")
    print(f"F1-score:   {metrics['f1']:.6f}")


if __name__ == "__main__":
    main()