# Finance MLOps Pipeline 🚀

This project is an end-to-end **MLOps pipeline** designed to predict Apple (AAPL) stock prices. It leverages industry-standard tools to ensure data quality, monitor model performance, and automate the entire machine learning lifecycle.

## 🏗️ Architecture

The system is built on a microservices architecture using **Docker**:

*   **Orchestrator:** [Apache Airflow](https://airflow.apache.org/) manages the workflow DAGs.
*   **Experiment Tracking:** [MLflow](https://mlflow.org/) logs metrics, parameters, and acts as the Model Registry.
*   **Database:** PostgreSQL (backend for Airflow).
*   **Data Validation:** [Pandera](https://pandera.readthedocs.io/) ensures data schema conformity.
*   **Drift Detection:** [Evidently AI](https://www.evidentlyai.com/) monitors data stable/drift status.
*   **Model:** PyTorch-based Time Series Transformer.

### Pipeline Flow

```mermaid
graph LR
    A[Data Ingestion] --> B[Validation (Pandera)]
    B --> C[Drift Detection (Evidently)]
    C --> D{Drift?}
    D -- Yes --> E[Retrain Model]
    D -- No --> F[Skip Training]
    E --> G[Predict Next Day]
    F --> G
    G --> H[Email Notification]
```

## 🛠️ Components & Technologies

| Stage | Tool | Description |
| :--- | :--- | :--- |
| **Ingestion** | `yfinance` | Downloads 2 years of hourly/daily AAPL data. |
| **Validation** | `Pandera` | Checks for positive prices and non-null values. Terminates pipeline on failure. |
| **Monitoring** | `Evidently` | Compares recent data vs. training data. Triggers retraining if drift > threshold. |
| **Training** | `PyTorch` | Transformer model (Encoder-Decoder) optimized with **Optuna**. Logs to MLflow. |
| **Prediction** | `PyTorch` | Loads the "Latest" champion model from MLflow Registry for inference. |
| **Alerting** | `SMTP` | Sends daily email with prediction and drift report attachments. |

## 🚀 Setup & Installation

### Prerequisites
*   Docker & Docker Compose
*   A Gmail account (for notifications) with an [App Password](https://support.google.com/accounts/answer/185833).

### 1. Clone & Configure
Clones the repository and navigate to the project folder.

### 2. Configure Credentials
Update `dags/notify.py` with your email credentials:
```python
SENDER_EMAIL = "your_email@gmail.com"
SENDER_PASSWORD = "your_app_password"
```

### 3. Build & Run
Start the infrastructure:
```bash
docker-compose up -d --build
```

## 💻 Usage

1.  **Access Airflow:** Go to `http://localhost:8080` (User/Pass: `admin`/`admin`).
2.  **Access MLflow:** Go to `http://localhost:5000` to view experiments and models.
3.  **Trigger DAG:**
    *   Enable the `finance_mlops_pipeline_v5_smart` DAG.
    *   Trigger it manually or wait for the daily schedule.

## 📂 Project Structure

```text
├── dags/
│   ├── finance_dag.py        # Main Airflow DAG definition
│   ├── prepare_data.py       # Data download (yfinance)
│   ├── validate_data.py      # Schema validation (Pandera)
│   ├── detect_drift.py       # Drift calculation (Evidently)
│   ├── train_model_script.py # Training logic & HPO (Optuna/MLflow)
│   ├── predict.py            # Inference logic
│   └── notify.py             # Email notification script
├── docker-compose.yaml       # Infrastructure definition
├── Dockerfile                # Custom Airflow image with dependencies
└── requirements.txt          # Python dependencies
```

## 📊 Key Features
*   **Smart Retraining:** The model is only retrained when **Module Drift** is detected, saving computational resources.
*   **Quality Gates:** The pipeline fails fast if data is corrupted (Pandera), implementation fails manually.
*   **Versioning:** All data and models are versioned in MLflow.
