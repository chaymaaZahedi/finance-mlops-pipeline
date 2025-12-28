
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import os
import json
from datetime import datetime

DATA_PATH = "/opt/airflow/dags/training_dataset.csv"
now = datetime.now().strftime("%Y-%m-%d_%Hh%M")
# Chemins d'archivage (Timestampés)
REPORT_PATH = f"/opt/airflow/dags/report/html/drift_report_{now}.html"
JSON_REPORT_PATH = f"/opt/airflow/dags/report/json/drift_report_{now}.json"

# Chemins pour le Pipeline (Fixes)
LATEST_REPORT_PATH = "/opt/airflow/dags/latest_drift_report.html"
LATEST_JSON_PATH = "/opt/airflow/dags/latest_drift_report.json"

def detect_drift():
    print("--- 🔎 DÉMARRAGE : DÉTECTION DE DRIFT ---")
    
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Le fichier de données entraînement n'existe pas : {DATA_PATH}")

    # Chargement
    df = pd.read_csv(DATA_PATH)
    
    # On split le dataset en deux pour simuler (Reference = Passé, Current = Récent)
    # Dans la vraie vie, Reference = Dataset d'entrainement validé, Current = Nouvelles données
    split_index = int(len(df) * 0.70)
    
    reference_data = df.iloc[:split_index]
    current_data = df.iloc[split_index:]
    
    print(f"Taille Ref: {len(reference_data)}, Taille Current: {len(current_data)}")

    # Création du rapport Evidently
    report = Report(metrics=[
        DataDriftPreset(), 
    ])

    report.run(reference_data=reference_data, current_data=current_data)

    # Sauvegarde HTML (Archive + Latest)
    report.save_html(REPORT_PATH)
    report.save_html(LATEST_REPORT_PATH)
    print(f"✅ Rapport HTML généré : {REPORT_PATH} & {LATEST_REPORT_PATH}")
    
    # Sauvegarde JSON (Archive + Latest)
    report.save_json(JSON_REPORT_PATH)
    report.save_json(LATEST_JSON_PATH)
    print(f"✅ Rapport JSON généré : {JSON_REPORT_PATH} & {LATEST_JSON_PATH}")

    # Vérification rapide (optionnel)
    # Vérification rapide (optionnel)
    with open(JSON_REPORT_PATH, 'r') as f:
        data = json.load(f)
        # 'drift_share' dans le résultat est le SEUIL. 'share_of_drifted_columns' est la valeur CALCULÉE.
        drift_threshold = data['metrics'][0]['result']['drift_share']
        actual_drift = data['metrics'][0]['result']['share_of_drifted_columns']
        dataset_drift = data['metrics'][0]['result']['dataset_drift']
        
        print(f"📊 Part de drift calculée : {actual_drift} (Seuil: {drift_threshold})")
        print(f"🚨 Drift Global détecté ? : {dataset_drift}")

if __name__ == "__main__":
    detect_drift()
