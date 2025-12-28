# from airflow import DAG
# from airflow.operators.bash import BashOperator
# from airflow.operators.python import PythonOperator
# from datetime import datetime, timedelta
# import yfinance as yf
# import pandas as pd
# import pandera as pa

# # --- Fonction pour la donnée du jour (Inférence future) ---
# # On la garde ici car elle est légère, ou on pourrait aussi la mettre dans un script à part.
# def extract_daily_data():
#     print("--- EXTRACTION DATA DU JOUR (POUR PRÉDICTION) ---")
#     data = yf.download("AAPL", period="1d", interval="1h")
#     if isinstance(data.columns, pd.MultiIndex):
#         data.columns = data.columns.get_level_values(0)
    
#     file_path = f"/opt/airflow/dags/aapl_data_{datetime.now().strftime('%Y%m%d')}.csv"
#     data[['Close']].to_csv(file_path)
#     print(f"✅ Données du jour sauvegardées : {file_path}")

# # --- DÉFINITION DU DAG ---
# default_args = {
#     'owner': 'moi',
#     'retries': 0,
#     'retry_delay': timedelta(minutes=5),
# }

# with DAG(
#     dag_id='finance_mlops_pipeline_v4_modular',
#     default_args=default_args,
#     description='Pipeline Modulaire : Prepare(Script) -> Train(Script) -> Daily',
#     start_date=datetime(2023, 1, 1),
#     schedule_interval='@daily',
#     catchup=False
# ) as dag:

#     # TÂCHE 1 : Lancer le script de téléchargement isolé
#     task_prepare_data = BashOperator(
#         task_id='prepare_data_script',
#         bash_command='python /opt/airflow/dags/prepare_data.py'
#     )

#     # 2. Détection de Drift (NOUVEAU)
#     task_detect_drift = BashOperator(
#         task_id='detect_data_drift',
#         bash_command='python /opt/airflow/dags/detect_drift.py'
#     )

#     # TÂCHE 3 : Lancer le script d'entraînement (qui lit le fichier créé par la Tâche 1)
#     task_train_model = BashOperator(
#         task_id='train_model_script',
#         bash_command='python /opt/airflow/dags/train_model_script.py'
#     )

#     # TÂCHE 4 : Extraction quotidienne (Python interne)
#     task_daily_extract = PythonOperator(
#         task_id='extract_daily',
#         python_callable=extract_daily_data
#     )

#     # TÂCHE 5 : PRÉDICTION (Le résultat final)
#     task_predict = BashOperator(
#         task_id='make_prediction',
#         bash_command='python /opt/airflow/dags/predict.py'
#     )

#     # --- ENCHAÎNEMENT ---
#     # 1. On prépare la donnée master
#     # 2. Une fois fini, on lance drift et entrainement
#     task_prepare_data >> [task_detect_drift, task_train_model]
#     task_train_model >> [task_daily_extract, task_predict]

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule
from datetime import datetime, timedelta
import json
import os

# --- FONCTION DE DÉCISION (L'Aiguilleur) ---
def check_drift_result():
    JSON_PATH = "/opt/airflow/dags/latest_drift_report.json"
    print(f"🔍 Lecture du rapport : {JSON_PATH}")
    
    try:
        if not os.path.exists(JSON_PATH):
            print("⚠️ Fichier introuvable. Par sécurité -> Entraînement.")
            return 'train_model_script'
            
        with open(JSON_PATH, 'r') as f:
            report = json.load(f)

        # --- CORRECTION ICI ---
        # On essaie de trouver la clé au bon endroit pour Evidently v0.7+
        try:
            # Chemin : metrics -> premier élément (0) -> result -> dataset_drift
            drift_detected = report['metrics'][0]['result']['dataset_drift']
        except (KeyError, IndexError, TypeError):
            # Si le chemin échoue, on regarde si c'est à la racine (anciennes versions)
            # ou on met False par défaut
            print("⚠️ Structure JSON inattendue, tentative racine...")
            drift_detected = report.get("drift_detected", False)

        print(f"🧐 Valeur lue dans le JSON : {drift_detected}")
        
        if drift_detected:
            print("🚨 DRIFT DÉTECTÉ ! -> Direction : Entraînement")
            return 'train_model_script'
        else:
            print("✅ Pas de Drift. -> Direction : Repos")
            return 'no_drift_task'
            
    except Exception as e:
        print(f"❌ Erreur lecture JSON ({e}). Par sécurité -> Entraînement.")
        return 'train_model_script'

# --- CONFIGURATION DU DAG ---
default_args = {
    'owner': 'mlops_engineer',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='finance_mlops_pipeline_v5_smart', # J'ai changé le nom pour la V5
    default_args=default_args,
    description='Pipeline : Prepare -> Detect Drift -> [Train IF Drift] -> Predict',
    start_date=datetime(2023, 1, 1),
    schedule_interval='@daily',
    catchup=False
) as dag:

    # 1. PRÉPARATION (Télécharge les 2 dernières années + Ajout de la journée d'hier)
    task_prepare = BashOperator(
        task_id='prepare_data',
        bash_command='python /opt/airflow/dags/prepare_data.py'
    )

    # 1.5 VALIDATION (Pandera)
    task_validate = BashOperator(
        task_id='validate_data_quality',
        bash_command='python /opt/airflow/dags/validate_data.py'
    )

    # 2. DÉTECTION DU DRIFT (Génère le JSON)
    task_detect = BashOperator(
        task_id='detect_drift',
        bash_command='python /opt/airflow/dags/detect_drift.py'
    )

    # 3. BRANCHING (L'Aiguilleur)
    task_check_branch = BranchPythonOperator(
        task_id='check_drift_threshold',
        python_callable=check_drift_result
    )

    # 4A. ENTRAÎNEMENT (Se lance seulement si l'aiguilleur le dit)
    task_train = BashOperator(
        task_id='train_model_script',
        bash_command='python /opt/airflow/dags/train_model_script.py'
    )

    # 4B. RIEN À FAIRE (Tâche vide pour boucher le trou si pas de drift)
    task_no_drift = EmptyOperator(
        task_id='no_drift_task'
    )

    # 5. PRÉDICTION (Le Final)
    # TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS est CRUCIAL ici.
    # Cela signifie : "Lance-toi même si 'train_model' a été sauté (skipped)."
    task_predict = BashOperator(
        task_id='make_prediction',
        bash_command='python /opt/airflow/dags/predict.py',
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS
    )

    # --- NOUVEAU : 6. NOTIFICATION EMAIL ---
    task_email = BashOperator(
        task_id='send_email_notification',
        bash_command='python /opt/airflow/dags/notify.py'
    )

    # --- LE CÂBLAGE LOGIQUE ---
    
    # Étape 1 à 1.5 à 2
    task_prepare >> task_validate >> task_detect
    
    # Étape 2 à 3 (Décision)
    task_detect >> task_check_branch
    
    # Étape 3 vers 4A ou 4B (Le "Y")
    task_check_branch >> [task_train, task_no_drift]
    
    # Étape 4A et 4B se rejoignent vers 5 (Convergence)
    task_train >> task_predict
    task_no_drift >> task_predict

    # On ajoute l'email après la prédiction
    task_predict >> task_email