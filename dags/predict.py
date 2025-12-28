import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import math
import os
from sklearn.preprocessing import MinMaxScaler
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

# --- CONFIGURATION ---
MLFLOW_URI = "http://mlflow-server:5000"
DATA_PATH = "/opt/airflow/dags/training_dataset.csv"
SEQ_LENGTH = 60 
# Le nom exact que vous avez mis dans registered_model_name
MODEL_NAME = "Apple_Transformer_Production"

# ... (Copiez-collez ici les classes PositionalEncoding et TimeSeriesTransformer comme avant) ...
# ... (Elles sont obligatoires pour PyTorch) ...
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.model_type = 'Transformer'
        self.encoder = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(d_model, 1)
        self.init_weights()
    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
    def forward(self, src):
        src = src.transpose(0, 1) 
        src = self.encoder(src) * math.sqrt(src.size(2))
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output[-1, :, :])
        return output
# ---------------------------------------------------------------------------

def predict_next_day():
    mlflow.set_tracking_uri(MLFLOW_URI)
    
    # --- LA MAGIE DU REGISTRY EST ICI ---
    # Juste avant de charger le modèle
    client = MlflowClient()
    # On récupère les infos de la dernière version du modèle
    latest_versions = client.get_latest_versions(MODEL_NAME)
    # On prend la version la plus récente
    current_version_obj = sorted(latest_versions, key=lambda x: int(x.version))[-1]
    current_version_num = current_version_obj.version
    
    print(f"ℹ️ Version du modèle utilisée : v{current_version_num}")
    # Plus besoin de chercher les ID, on demande la dernière version
    model_uri = f"models:/{MODEL_NAME}/Latest"
    
    print(f"📥 Chargement du modèle depuis le Registry : {model_uri}")
    try:
        model = mlflow.pytorch.load_model(model_uri)
    except Exception as e:
        print("❌ Erreur : Impossible de charger le modèle. Avez-vous bien entraîné le modèle avec 'registered_model_name' ?")
        raise e
        
    model.eval()
    
    # 2. Charger les données (Le reste est identique)
    print(f"📂 Lecture des données : {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    
    if 'Close' in df.columns:
        data = df[['Close']].values
    else:
        data = df.iloc[:, -1].values.reshape(-1, 1)
        
    # Scaling
    scaler = MinMaxScaler()
    scaler.fit(data) 
    
    # 60 derniers jours
    last_60_days = data[-SEQ_LENGTH:]
    last_60_days_scaled = scaler.transform(last_60_days)
    input_tensor = torch.Tensor(last_60_days_scaled).unsqueeze(0)
    
    # Prédiction
    with torch.no_grad():
        prediction_scaled = model(input_tensor)
        
    prediction_price = scaler.inverse_transform(prediction_scaled.numpy())
    final_price = prediction_price[0][0]
    last_real_price = data[-1][0]
    
    print("=" * 40)
    print(f"📅 DATE DU RUN : {pd.Timestamp.now()}")
    print(f"📉 PRIX HIER    : {last_real_price:.2f} $")
    print(f"🔮 PRÉDICTION   : {final_price:.2f} $")
    print("=" * 40)

    # On crée un fichier qui stocke l'historique de nos prédictions
    history_path = "/opt/airflow/dags/prediction_history.csv"
    
    # On prépare la ligne à ajouter
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    new_row = pd.DataFrame({
        "Date_Run": [timestamp],
        "Last_Real_Price": [last_real_price],
        "Predicted_Price": [final_price],
        "Model_Version": [current_version_num]
    })
    
    # Mode 'append' (ajout) : si le fichier existe, on ajoute à la fin. Sinon on le crée.
    if not os.path.exists(history_path):
        new_row.to_csv(history_path, index=False)
    else:
        new_row.to_csv(history_path, mode='a', header=False, index=False)
        
    print(f"✅ Prédiction sauvegardée dans : {history_path}")

if __name__ == "__main__":
    predict_next_day()