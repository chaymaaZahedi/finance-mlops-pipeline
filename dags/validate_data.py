
import pandera as pa
from pandera import Column, Check, DataFrameSchema
import pandas as pd
import os

# Chemins
RAW_DATA_PATH = "/opt/airflow/dags/training_dataset.csv"

# --- DÉFINITION DU SCHÉMA PANDERA ---
# On définit ici les règles que nos données doivent STRICTEMENT respecter.
finance_schema = DataFrameSchema({
    "Open": Column(float, checks=[
        Check.ge(0, error="Le prix d'ouverture doit être positif"), 
    ], nullable=False),
    "High": Column(float, checks=Check.ge(0)),
    "Low": Column(float, checks=Check.ge(0)),
    "Close": Column(float, checks=Check.ge(0)),
    "Volume": Column(float, checks=Check.ge(0)),  # Volume peut être float ou int selon Yahoo
}, coerce=True) # <--- IMPORTANT: Convertit automatiquement les types (ex: int -> float)

def validate_data():
    print("--- 🛡️ DÉMARRAGE : VALIDATION PANDERA ---")
    
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"❌ Données introuvables : {RAW_DATA_PATH}")
    
    # 1. Chargement
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"📄 Données chargées : {len(df)} lignes.")

    # 2. Validation
    try:
        validated_df = finance_schema.validate(df)
        print("✅ SUCCÈS : Les données respectent le schéma Pandera/Finance !")
        return True
    except pa.errors.SchemaError as exc:
        print("🚨 ÉCHEC : Validation Pandera a échoué !")
        print(f"❌ Erreur : {exc}")
        # On peut choisir de casser le pipeline ici :
        raise ValueError("Les données ne sont pas conformes au standard de qualité.")

if __name__ == "__main__":
    validate_data()
