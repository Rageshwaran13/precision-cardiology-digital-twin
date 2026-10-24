# Precision Cardiology Digital Twin

**An AI-powered cardiac monitoring research prototype combining synthetic electronic health records, wearable physiological signals, machine learning, and interactive 3D heart visualization.**

**Project category:** Healthcare AI / Digital Health / Machine Learning

> **Research demonstration only.** This project uses synthetic data and is not intended for diagnosis, treatment, emergency response, or clinical decision-making. Its model performance does not establish clinical accuracy.

## 1. Team Details

| S. No. | Team Member   |
| ------ | ------------- |
| 1      | Rageshwaran P |
| 2      | Laveena M     |
| 3      | Yuvasri D S   |
| 4      | Yashwanth D   |

## 2. College / Incubator Information

* **College / Institution:** [Enter your official college name]
* **Department:** [Enter your department]
* **Incubator / Innovation Centre:** [Enter the official name, or state "Not applicable" if permitted by the submission rules]

Replace these placeholders with your actual institutional details before submission.

## 3. Problem Statement

Cardiac conditions can involve changes in physiological signals before an event becomes clinically apparent. Conventional monitoring systems may present historical records and current measurements without combining them into a unified, forward-looking risk visualization.

This project explores a research prototype that combines patient-level clinical information with wearable-style physiological time-series data to demonstrate how machine learning could identify synthetic cardiac-anomaly patterns.

The objective is to integrate these data sources into an interactive dashboard that presents risk estimates, physiological trends, and a 3D heart visualization in one interface.

## 4. Healthcare Use Case

The prototype demonstrates a potential clinical decision-support workflow for research into cardiac monitoring.

Its intended demonstration workflow is:

1. Review a simulated patient's demographic and clinical information.
2. Examine wearable-style measurements such as heart rate, heart-rate variability, oxygen saturation, and sleep-related signals.
3. Process physiological measurements using engineered time-series features.
4. Display the model-generated cardiac-anomaly risk estimate.
5. Explore the associated information through a dashboard with an interactive 3D heart visualization.

The system is intended to illustrate a possible future monitoring workflow. It has not been clinically validated and must not be used to make decisions about real patients.

## 5. Technical Stack

| Component             | Technology                                                 |
| --------------------- | ---------------------------------------------------------- |
| Programming language  | Python                                                     |
| Data processing       | Pandas, NumPy                                              |
| Machine learning      | scikit-learn, with XGBoost support where available         |
| Interactive dashboard | Streamlit                                                  |
| Data visualization    | Plotly                                                     |
| 3D visualization      | Three.js and WebGL                                         |
| Deployment            | Docker and Docker Compose                                  |
| Data                  | Synthetic EHR records and generated wearable-style signals |

## 6. AI/ML Model Details

The prototype uses a gradient-boosting classification pipeline to demonstrate cardiac-anomaly risk estimation.

### Dataset design

* 500 simulated patients.
* 30 days of hourly wearable-style records per patient.
* 720 hourly records per simulated patient.
* 360,000 wearable-style hourly records in total.
* 33 engineered model features.
* Synthetic anomaly patterns introduced into the final 24-hour window for a subset of simulated patients.

### Feature engineering

The pipeline combines clinical baseline information with wearable-style physiological measurements and six-hour rolling summaries.

The feature groups include:

* Clinical and demographic baseline features.
* Instantaneous wearable measurements.
* Rolling physiological statistics.
* Sleep-stage indicators.
* Encoded demographic variables.

### Model evaluation

The model is evaluated using a patient-level split in the synthetic demonstration dataset.

Reported metrics from this synthetic task must not be interpreted as evidence of real-world clinical performance. The generated anomaly patterns are intentionally separable, and further evaluation on appropriate real-world datasets, calibration, leakage assessment, and independent validation would be required before clinical applications could be considered.

## 7. System Architecture

The project follows this conceptual pipeline:

Synthetic EHR Generation + Wearable Signal Generation
→ Data Integration
→ Feature Engineering
→ Gradient-Boosting Classification
→ Risk Estimate and Physiological Trends
→ Streamlit Dashboard + Interactive 3D Heart

The architecture documentation and project presentation are provided in the repository's `docs/` directory.

## 8. Demo Video

**Working demonstration video:** [in process]

The demonstration should show the application running, patient information, physiological charts, the risk-prediction interface, and the interactive 3D heart.

The video must be uploaded to YouTube as **Unlisted**, and the link must be added here before final submission.

## 9. Architecture and Project Documents

* [Architecture / Project Blueprint PDF](docs/precision_cardiology_blueprint.pdf)
* [Project Presentation PowerPoint](docs/Precision_Cardiology_Digital_Twin_Presentation.pptx)

Verify that the blueprint contains the architecture diagram required by the submission checklist. If it does not, add a separate architecture diagram and export it as PDF or PPT.

If the submission requires both PDF and PPT presentation formats, export the existing PowerPoint to PDF and add the resulting file to `docs/`.

## 10. Installation and Execution

### Prerequisites

* Python 3.11 recommended.
* Git.
* Sufficient disk space for the generated datasets.

### Install dependencies

```bash
python -m venv .venv
pip install -r requirements.txt
```

### Generate synthetic data and train the model

```bash
python scripts/setup_demo.py
```

### Run the dashboard

```bash
streamlit run dashboard/app.py
```

Open the local address displayed by Streamlit, normally `http://localhost:8501`.

See the existing setup instructions in this README for environment-specific commands.

## 11. Open-Source License

Third-party libraries, 3D models, and other external assets may have separate license requirements. Their original notices and redistribution conditions must be respected.

A project-wide license must not be assumed to cover third-party materials that the team does not own.

## 12. Limitations and Future Work

This is a proof-of-concept built using synthetic data. Its generated patterns and evaluation metrics do not establish clinical performance.

Future work includes:

* Evaluation using appropriately governed and licensed clinical datasets.
* Temporal validation and additional checks for information leakage.
* Probability calibration and uncertainty estimation.
* Explainable model outputs.
* Clinical usability evaluation and applicable regulatory review.

## 13. Safety and Privacy

The current demonstration uses synthetic patient data. No real patient records should be committed to this public repository.

The project is not a medical device and must not be used to guide diagnosis, treatment, or patient care.

## 14. Repository

**Public GitHub repository:**

https://github.com/Rageshwaran13/precision-cardiology-digital-twin
