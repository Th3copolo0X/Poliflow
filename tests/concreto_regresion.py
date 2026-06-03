import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from poliflow.core.Tensor import Tensor
from poliflow.nn.Activacion import ReLU
from poliflow.nn.Lineal import Lineal
from poliflow.nn.Secuencial import Secuencial
from poliflow.optim.SGD import SGD
from poliflow.perdidas.MSE import MSE


FEATURES = [
    "cement",
    "blast_furnace_slag",
    "fly_ash",
    "water",
    "superplasticizer",
    "coarse_aggregate",
    "fine_aggregate ",
    "age",
]
TARGET = "concrete_compressive_strength"


def train_test_split_np(x, y, test_size=0.2, seed=123):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(x))
    rng.shuffle(idx)
    n_test = int(round(len(x) * test_size))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]


def regression_metrics(y_true, y_pred):
    residual = y_pred - y_true
    mse = float((residual ** 2).mean())
    rmse = float(np.sqrt(mse))
    mae = float(np.abs(residual).mean())
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot else 0.0
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


def main():
    np.random.seed(0)

    data_path = ROOT / "tests" / "data" / "concrete_data.csv"
    df = pd.read_csv(data_path)

    x = df[FEATURES].to_numpy(dtype=float)
    y = df[[TARGET]].to_numpy(dtype=float)

    x_train, x_test, y_train_orig, y_test_orig = train_test_split_np(
        x, y, test_size=0.2, seed=123
    )

    x_mean = x_train.mean(axis=0, keepdims=True)
    x_std = x_train.std(axis=0, keepdims=True)
    x_std[x_std == 0] = 1.0
    x_train = (x_train - x_mean) / x_std
    x_test = (x_test - x_mean) / x_std

    y_mean = y_train_orig.mean(axis=0, keepdims=True)
    y_std = y_train_orig.std(axis=0, keepdims=True)
    y_std[y_std == 0] = 1.0
    y_train = (y_train_orig - y_mean) / y_std
    y_test = (y_test_orig - y_mean) / y_std

    model = Secuencial(
        Lineal(8, 16),
        ReLU(),
        Lineal(16, 8),
        ReLU(),
        Lineal(8, 1),
    )
    criterion = MSE()
    optimizer = SGD(model.parameters(), lr=0.01)

    x_train_t = Tensor(x_train)
    y_train_t = Tensor(y_train)

    epochs = 1000
    start = time.perf_counter()
    for _ in range(epochs):
        y_pred = model(x_train_t)
        loss = criterion(y_pred, y_train_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    elapsed = time.perf_counter() - start

    train_pred_norm = model(Tensor(x_train)).data
    test_pred_norm = model(Tensor(x_test)).data

    train_loss = float(((train_pred_norm - y_train) ** 2).mean())
    test_loss = float(((test_pred_norm - y_test) ** 2).mean())

    test_pred_orig = test_pred_norm * y_std + y_mean
    metrics = regression_metrics(y_test_orig, test_pred_orig)

    print("PoliFlow - Regresion concrete")
    print(f"Train Loss: {train_loss:.6f}")
    print(f"Test Loss:  {test_loss:.6f}")
    print(f"Tiempo (s): {elapsed:.4f}")
    print(f"MSE:        {metrics['mse']:.6f}")
    print(f"RMSE:       {metrics['rmse']:.6f}")
    print(f"MAE:        {metrics['mae']:.6f}")
    print(f"R2:         {metrics['r2']:.6f}")


if __name__ == "__main__":
    main()