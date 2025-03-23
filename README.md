# 🔊 DCASE2024 ASD Project – Anomalous Sound Detection

This project is a complete implementation of **DCASE 2024 - Task 2**, aimed at detecting abnormal sounds emitted by industrial machines.

---

## 📌 Goals

- Detection of audio anomalies using **unsupervised learning**.
- Training a **autoencoder** on normal sounds
- Identification of anomalies by **reconstruction error**.
- Generation of reports and submission files in **DCASE** format

---

## 📁 Project structure

| Dossier         | Contents                                         |
|-----------------|--------------------------------------------------|
| `notebooks/`    | 📒 EDA, training, inference                      |
| `src/`          | 📦 Source code (config, extraction, models...)   |
| `data/`         | 📂 Preprocessed data, models, thresholds         |
| `outputs/`      | 📊 Predictions and reports                       |
| `logs/`         | 📝 Execution logs                                |

---

## ⚙️ Main steps

### 1. Exploratory Data Analysis (EDA)
→ `notebooks/01_EDA_audio_features.ipynb`

- Signal visualization (waveforms, spectrograms, MFCCs)
- Feature extraction and statistical analysis

### 2. Autoencoder model training
→ `notebooks/02_model_autoencoder_training.ipynb`

- Training on normal sounds
- Save model, scaler, threshold

### 3. inference and detection on new files
→ `notebooks/03_inference_prediction.ipynb`

- Apply model to test sounds
- Report generation and submission

---

## 📊 Generated files

| File                                   | Description                                |
|----------------------------------------|--------------------------------------------|
| `autoencoder_model.h5`                 | Autoencoder model driven                   |
| `scaler.pkl`                           | StandardScaler for normalization           |
| `threshold.npy`                        | Anomaly threshold (MSE)                    |
| `challenge_predictions.csv`            | Results file by file                       |
| `report_by_machine.csv`                | Aggregate report by machine                |
| `dcase_submission.csv`                 | DCASE format for submission                |

---

## 🚀 Fast execution

```bash
# Prepare the environment
pip install -r requirements.txt

# Run notebooks in order:
1. 01_EDA_audio_features.ipynb
2. 02_model_autoencoder_training.ipynb
3. 03_inference_prediction.ipynb
