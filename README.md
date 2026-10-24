# Precision Cardiology Digital Twin

A research and hackathon proof-of-concept that combines synthetic electronic health record (EHR) data with wearable-style time-series signals and presents a model-generated cardiac-anomaly risk estimate in an interactive Streamlit dashboard with a 3D heart visualization.

> **Research demonstration only. Not for diagnosis, treatment, emergency response, or clinical decision-making.** The data are synthetic, and model results must not be interpreted as evidence of real-world clinical performance.

![Digital twin heart visualization](dashboard/static/digital_twin_heart.jpg)

## Project overview

The pipeline is designed to demonstrate five steps:

1. **Generate synthetic EHR records** for 500 simulated patients.
2. **Generate wearable-style time series** with 720 hourly records per patient (30 days).
3. **Merge and engineer features**, including six-hour rolling summaries.
4. **Train a classifier** using the generated dataset and save the model artifacts locally.
5. **Explore a Streamlit clinical cockpit** with EHR summaries, telemetry charts, a forecast banner, and an interactive 3D heart.

The synthetic generator injects anomaly patterns into the final 24-hour window for a subset of simulated patients. This is useful for testing the software pipeline, but makes the task substantially easier than real clinical forecasting. Near-perfect scores on this generated data do **not** demonstrate clinical accuracy.

## Technology

- **Python** for data generation and model training
- **Pandas / NumPy / scikit-learn**, with XGBoost support where available
- **Streamlit / Plotly** for the dashboard and telemetry charts
- **Three.js / WebGL** for the 3D heart visualization
- **Docker Compose** for containerized dashboard execution

## Repository layout

```text
precision-cardiology-digital-twin/
├── dashboard/
│   ├── app.py
│   ├── components/                 # EHR panel, risk banner, heart widget
│   └── static/                     # Three.js assets and 3D heart model
├── data/
│   ├── raw/                        # Generated locally; not committed
│   └── processed/                  # Generated locally; not committed
├── docs/
│   ├── Precision_Cardiology_Digital_Twin_Presentation.pptx
│   └── precision_cardiology_blueprint.pdf
├── models/                         # Generated locally; not committed
├── scripts/
│   └── setup_demo.py               # Rebuild data and train the model
├── src/
│   ├── data/                       # Synthetic data generation and merging
│   ├── model/                      # Feature engineering, training, inference
│   └── utils/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

Generated CSV datasets and trained model artifacts are intentionally excluded from Git so the repository stays easier to clone and review. The setup script recreates them on your machine.

## Run locally

### 1. Install Python

Use Python **3.11** for the most predictable setup.

### 2. Create and activate a virtual environment

**Windows PowerShell**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Generate the demonstration data and train the model

From the repository root, run:

```bash
python scripts/setup_demo.py
```

This runs the EHR generator, wearable generator, dataset merge, and model training in order. The generated time-series and merged CSV files can be large, so keep sufficient disk space available. The files are created under `data/` and the trained artifacts under `models/`.

### 5. Start the dashboard

```bash
streamlit run dashboard/app.py
```

Open the local URL printed by Streamlit (normally `http://localhost:8501`).

## Run with Docker

First complete steps 1–4 above on the host machine so `data/` and `models/` contain the generated files. Then run:

```bash
docker compose up --build
```

Open `http://localhost:8501`. The Compose configuration mounts the local `data/`, `models/`, `dashboard/`, and `src/` directories into the container.

To stop the container:

```bash
docker compose down
```

## Rebuilding the data

To regenerate the whole demo dataset and retrain the model:

```bash
python scripts/setup_demo.py
```

The generated CSV and model files are ignored by Git. Do not commit real patient data, secrets, API keys, or any other sensitive information.

## Evaluation and limitations

The evaluation values produced by the training script apply to a **synthetic, intentionally separable demonstration task**. They are not clinical validation results. Before any clinical use could even be considered, the project would require, at minimum:

- Evaluation on appropriately licensed, representative clinical data with suitable governance and approvals
- Temporal and patient-level validation designed to avoid information leakage
- Calibration, robust uncertainty reporting, and independent external validation
- Clinical review, usability testing, privacy/security assessment, and applicable regulatory review

The project does not provide medical advice and must not be used to guide patient care.

## Team

- Rageshwaran P
- Laveena M
- Yuvasri D S
- Yashwanth D

## Project documents

The presentation deck and project blueprint are available in [`docs/`](docs/). A step-by-step publishing guide is in [`docs/GITHUB_UPLOAD.md`](docs/GITHUB_UPLOAD.md).

## Before making the repository public

Check that you have permission to redistribute every non-code asset, particularly `dashboard/static/heart_model.glb` and `dashboard/static/digital_twin_heart.jpg`. The bundled Three.js library identifies itself as MIT-licensed; preserve upstream notices when redistributing it. See [`docs/THIRD_PARTY_ASSETS.md`](docs/THIRD_PARTY_ASSETS.md).

No project-wide open-source license has been selected in this starter repository. Choose a license deliberately before inviting reuse or contributions.
