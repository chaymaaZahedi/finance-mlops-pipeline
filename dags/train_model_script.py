import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import math
import os
from sklearn.preprocessing import MinMaxScaler
import mlflow
import mlflow.pytorch
import optuna
from datetime import datetime

# =========================
# CONFIGURATION
# =========================
MLFLOW_URI = "http://mlflow-server:5000"
# EXPERIMENT_NAME = f"Apple_Train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
EXPERIMENT_NAME = "Apple_Transformer_Optimization_V3"
run_date = datetime.now().strftime("%Y-%m-%d_%Hh%M")

DATA_PATH = "/opt/airflow/dags/training_dataset.csv"
N_TRIALS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# POSITIONAL ENCODING
# =========================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(1))

    def forward(self, x):
        return x + self.pe[:x.size(0)]


# =========================
# TRANSFORMER CAUSAL
# =========================
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, dropout):
        super().__init__()

        self.encoder = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=False
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.decoder = nn.Linear(d_model, 1)

    def _causal_mask(self, size, device):
        return torch.triu(torch.full((size, size), float("-inf")), diagonal=1).to(device)

    def forward(self, x):
        # x: (batch, seq, features)
        x = x.transpose(0, 1)  # (seq, batch, features)
        x = self.encoder(x)
        x = self.pos_encoder(x)

        mask = self._causal_mask(x.size(0), x.device)
        x = self.transformer(x, mask)

        return self.decoder(x[-1])


# =========================
# DATA LOADING (NO LEAKAGE)
# =========================
def load_data(seq_length):
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} introuvable")

    df = pd.read_csv(DATA_PATH)
    values = df[['Close']].values if 'Close' in df.columns else df.iloc[:, -1].values.reshape(-1, 1)

    # Chronological split
    train_end = int(len(values) * 0.7)
    val_end = int(len(values) * 0.85)

    train_raw = values[:train_end]
    val_raw = values[train_end:val_end]
    test_raw = values[val_end:]

    # Scaler FIT ONLY on train
    scaler = MinMaxScaler()
    scaler.fit(train_raw)

    train_scaled = scaler.transform(train_raw)
    val_scaled = scaler.transform(val_raw)
    test_scaled = scaler.transform(test_raw)

    def make_sequences(data):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length])
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    X_train, y_train = make_sequences(train_scaled)
    X_val, y_val = make_sequences(val_scaled)
    X_test, y_test = make_sequences(test_scaled)

    return (
        X_train.to(DEVICE), y_train.to(DEVICE),
        X_val.to(DEVICE), y_val.to(DEVICE),
        X_test.to(DEVICE), y_test.to(DEVICE),
        scaler
    )


# =========================
# OPTUNA OBJECTIVE
# =========================
def objective(trial):
    params = {
        "d_model": trial.suggest_categorical("d_model", [32, 64, 128]),
        "nhead": trial.suggest_categorical("nhead", [2, 4]),
        "num_layers": trial.suggest_int("num_layers", 1, 3),
        "dropout": trial.suggest_float("dropout", 0.1, 0.3),
        "lr": trial.suggest_loguniform("lr", 1e-4, 1e-2),
        "epochs": 10,
        "seq_length": 60
    }

    if params["d_model"] % params["nhead"] != 0:
        raise optuna.TrialPruned()

    with mlflow.start_run(nested=True):
        mlflow.log_params(params)

        X_train, y_train, X_val, y_val, _, _, _ = load_data(params["seq_length"])

        if len(X_train) == 0 or len(X_val) == 0:
            return float("inf")

        model = TimeSeriesTransformer(
            1, params["d_model"], params["nhead"],
            params["num_layers"], params["dropout"]
        ).to(DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
        loss_fn = nn.MSELoss()

        for _ in range(params["epochs"]):
            model.train()
            optimizer.zero_grad()
            loss = loss_fn(model(X_train), y_train)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val), y_val)

        mlflow.log_metric("val_loss", val_loss.item())
        return val_loss.item()


# =========================
# MAIN
# =========================
def main():
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_TRIALS)

    best = study.best_params
    best["seq_length"] = 60
    run_name_dynamic = f"Champion_{run_date}"
    with mlflow.start_run(run_name=run_name_dynamic):
        mlflow.log_params(best)
        mlflow.log_artifact(DATA_PATH)

        X_train, y_train, _, _, X_test, y_test, _ = load_data(best["seq_length"])

        model = TimeSeriesTransformer(
            1, best["d_model"], best["nhead"],
            best["num_layers"], best["dropout"]
        ).to(DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=best["lr"])
        loss_fn = nn.MSELoss()

        for _ in range(20):
            optimizer.zero_grad()
            loss = loss_fn(model(X_train), y_train)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            test_loss = loss_fn(model(X_test), y_test)

        mlflow.log_metric("test_loss", test_loss.item())
        # mlflow.pytorch.log_model(model, "transformer_model")
        mlflow.pytorch.log_model(
            model, 
            "transformer_model",
            registered_model_name="Apple_Transformer_Production" # <--- LA CLEF
        )
        print(f"✅ Modèle final évalué sur TEST — Loss: {test_loss.item():.6f}")


if __name__ == "__main__":
    main()
