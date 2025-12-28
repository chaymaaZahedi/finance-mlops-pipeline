import smtplib
import pandas as pd
import os
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase # <--- Pour les pièces jointes
from email import encoders           # <--- Pour l'encodage
from datetime import datetime

# --- CONFIGURATION ---
SENDER_EMAIL = "Your email sender"
SENDER_PASSWORD = "Your password app"
RECEIVER_EMAIL = "Your email receiver"

# Chemins des fichiers
CSV_PATH = "/opt/airflow/dags/prediction_history.csv"
DRIFT_JSON_PATH = "/opt/airflow/dags/latest_drift_report.json"
DRIFT_HTML_PATH = "/opt/airflow/dags/latest_drift_report.html"

def send_notification():
    print("📧 Préparation de l'email...")
    
    # --- 1. RECUPERATION DONNEES PREDICTION ---
    if not os.path.exists(CSV_PATH):
        print("❌ Pas d'historique de prédiction trouvé.")
        return

    try:
        df = pd.read_csv(CSV_PATH)
        last_row = df.iloc[-1]
        
        last_price = float(last_row['Last_Real_Price'])
        predicted_price = float(last_row['Predicted_Price'])
        date_run = last_row['Date_Run']
        model_version = last_row.get('Model_Version', 'Inconnue')
        
        evolution = ((predicted_price - last_price) / last_price) * 100
        emoji_trend = "🚀" if evolution > 0 else "🔻"
        
    except Exception as e:
        print(f"❌ Erreur lecture CSV : {e}")
        return

    # --- 2. VERIFICATION DU DRIFT ---
    drift_detected = False
    drift_msg = "✅ Aucun Drift détecté (Données stables)."
    
    if os.path.exists(DRIFT_JSON_PATH):
        try:
            with open(DRIFT_JSON_PATH, 'r') as f:
                report = json.load(f)
                # On cherche la clé de drift (compatible Evidently v0.7+)
                try:
                    drift_detected = report['metrics'][0]['result']['dataset_drift']
                except (KeyError, IndexError):
                    drift_detected = report.get("drift_detected", False)
            
            if drift_detected:
                drift_msg = "🚨 ALERTE : DATA DRIFT DÉTECTÉ !"
        except Exception as e:
            print(f"⚠️ Impossible de lire le statut du drift : {e}")

    # --- 3. CONSTRUCTION DU MESSAGE ---
    subject = f"{emoji_trend} Prédiction Apple : {predicted_price:.2f} $ | Drift: {'OUI' if drift_detected else 'NON'}"
    
    body = f"""
    Bonjour,
    
    Voici le résultat du pipeline MLOps du {date_run}.
    
    📊 ETAT DU MARCHÉ (DATA DRIFT) :
    -----------------------------------
    {drift_msg}
    {'(Les rapports HTML et JSON sont joints à ce mail)' if drift_detected else ''}
    
    🔮 PRÉDICTION :
    -----------------------------------
    📉 Prix Précédent    : {last_price:.2f} $
    🔮 Prédiction Demain : {predicted_price:.2f} $
    📈 Tendance Prévue   : {evolution:+.2f} %
    
    🤖 Modèle utilisé : {model_version}
    (Nom technique : Apple_Transformer_Production)
    
    Cordialement,
    Votre Airflow Bot
    """

    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECEIVER_EMAIL
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    # --- 4. ATTACHEMENT DES FICHIERS (CONDITIONNEL) ---
    if drift_detected:
        print("📎 Drift détecté : Ajout des pièces jointes...")
        files_to_attach = [DRIFT_HTML_PATH, DRIFT_JSON_PATH]
        
        for file_path in files_to_attach:
            if os.path.exists(file_path):
                try:
                    # Ouverture en binaire
                    with open(file_path, "rb") as attachment:
                        part = MIMEBase("application", "octet-stream")
                        part.set_payload(attachment.read())
                    
                    # Encodage pour email
                    encoders.encode_base64(part)
                    
                    # Ajout des headers (Nom du fichier)
                    filename = os.path.basename(file_path)
                    part.add_header(
                        "Content-Disposition",
                        f"attachment; filename= {filename}",
                    )
                    msg.attach(part)
                    print(f"   -> Ajouté : {filename}")
                except Exception as e:
                    print(f"   -> Erreur ajout fichier {file_path}: {e}")
            else:
                print(f"   -> Fichier introuvable : {file_path}")

    # --- 5. ENVOI ---
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        text = msg.as_string()
        server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, text)
        server.quit()
        print(f"✅ Email envoyé avec succès (Drift: {drift_detected}) !")
    except Exception as e:
        print(f"❌ Erreur lors de l'envoi : {e}")

if __name__ == "__main__":
    send_notification()