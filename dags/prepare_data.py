import yfinance as yf
import pandas as pd
import os

DATA_PATH = "/opt/airflow/dags/training_dataset.csv"

def fetch_history():
    print("--- 📥 DÉBUT : TÉLÉCHARGEMENT HISTORIQUE COMPLET (2 ANS) ---")
    
    # 1. Téléchargement
    df = yf.download("AAPL", period="2y", interval="1d", auto_adjust=True)
    
    # 2. Nettoyage du MultiIndex (Problème Yahoo)
    if isinstance(df.columns, pd.MultiIndex):
        try:
            # On essaie d'aplatir proprement si c'est formaté "Price | Ticker"
            df.columns = df.columns.get_level_values(0)
        except:
            pass

    # 3. Sauvegarde de TOUT (Open, High, Low, Close, Volume)
    # On ne filtre plus. On garde tout le dataset brut.
    df.to_csv(DATA_PATH)
    
    print(f"✅ SUCCÈS : Dataset complet sauvegardé dans {DATA_PATH}")
    print(f"📊 Colonnes disponibles : {list(df.columns)}")
    print(f"📊 Nombre de lignes : {len(df)}")

if __name__ == "__main__":
    fetch_history()