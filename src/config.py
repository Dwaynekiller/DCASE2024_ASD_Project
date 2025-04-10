"""
config.py

Centralized configuration for audio anomaly detection (DCASE 2024 Task 2)

Defines:
- Directory structure (root, data, logs, models)
- Feature extraction parameters (FFT, MFCC, etc.)
- Paths for saving models, scalers, thresholds

Author: GUIDY Mike
Date: 2025-03-12
"""
from pathlib import Path
DEBUG_MODE = True  # 🔄 Passe à True pour le debug complet

## ==========================================
# 🧭 1. Localisation automatique du projet
# ==========================================

# Trouve automatiquement le dossier "Datascience" à partir du notebook courant
notebook_path = Path.cwd()
ROOT = None
for parent in notebook_path.parents:
    if parent.name == "Datascience":
        ROOT = parent
        break
if ROOT is None:
    raise FileNotFoundError("Base directory 'Datascience' not found!")

ENV = "_Dev"
ENV_NAME = "ASD_Project"+ENV
# ==========================================
# 🏗️ 2. Répertoires de base du projet
# ==========================================
# 📁 Dossier racine du projet DCASE
PROJECT_DIR = ROOT / "Projet" / "DCASE2024" / ENV_NAME

# 📁 Dossier racine du projet DCASE
PROJECT_DATA_DIR = PROJECT_DIR / "data"

# 📁 Répertoire contenant les jeux de données audio bruts
AUDIO_DATA_ROOT = ROOT / "Data" / "sound_datasets"

# 📁 Répertoire spécifique au dataset DCASE
DCASE_DATASET_DIR = AUDIO_DATA_ROOT / "DCASE_DATASET"

# ==========================================
# 🏗️ 3. Processed audio and features
# ==========================================
# 📁 Fichiers audio convertis en .npy (après normalisation)
DEV_AUDIO_DIR = DCASE_DATASET_DIR / "dev_data"

# 📁 Fichiers audio convertis en .npy (après normalisation)
NUMPY_DATA_DIR = DCASE_DATASET_DIR / "numpy_data"

# 📁 Vecteurs de features extraits depuis les fichiers .npy
FEATURES_DIR = DCASE_DATASET_DIR / "features"

# 📁 Répertoire pour les features sauvegarder X et y
HDF5_DIR = FEATURES_DIR / "raw"

# 📁 Répertoire pour les features sauvegarder X et y
PROCESSED_DIR = FEATURES_DIR / "processed"

# 📁 Vecteurs de features réduits (PCA etc..) dans des fichiers .npy
REDUCED_DIR = DCASE_DATASET_DIR / "reduced_features"

# ==========================================
# 🏗️ 4. Model output and logs
# ==========================================
# 📁 Répertoire où seront enregistrés les modèles entraînés
MODEL_DIR = PROJECT_DATA_DIR / "models"

# 📁 Répertoire où seront enregistrés les modèles entraînés
REPORTS_DIR = PROJECT_DATA_DIR / "reports"

# 📁 Répertoire où seront enregistrés les modèles entraînés
SCALER_DIR = PROJECT_DATA_DIR / "scalers"

# 📁 Répertoire pour les fichiers de prédictions / inférence
IMAGES_DIR = PROJECT_DATA_DIR / "outputs" / "images"

# 📁 Répertoire pour les fichiers de prédictions / inférence
PREDICTIONS_DIR = PROJECT_DATA_DIR / "outputs" / "predictions"

# 📁 Dossier pour les fichiers de logs
LOG_DIR = PROJECT_DIR / "logs"

# ==========================================
# 🏗️ 5. Metadata
# ==========================================
# === Feature Storage (NPZ) ===
NPZ_FEATURES_PATH = PROCESSED_DIR / "audio_features.npz"

# === Feature Storage (NPZ) ===
NPZ_REDUCED_FEATURES_PATH = PROCESSED_DIR / "reduced_features.npz"

# 📄 Fichier CSV des métadonnées (des fichiers .wav d'origine)
METADATA_FILE = DCASE_DATASET_DIR / "dev_data.csv"

# 📄 Fichier CSV des métadonnées des features extraits (des fichiers .h5 d'origine)
DFMETA_FILE_PATH = PROCESSED_DIR / "df_meta.csv"

# 📄 Fichier log principal
LOG_FILE = LOG_DIR / "data_processing.log"

# ==========================================
# 💾 6. Paths for model and threshold saving
# ==========================================
# 📄 Modèle Keras autoencodeur
MODEL_PATH = MODEL_DIR / "autoencoder_model.h5"

# 📄 Scaler (StandardScaler) utilisé pour normaliser les features
SCALER_PATH = MODEL_DIR / "scaler.pkl"

# 📄 Seuil optimal d'anomalie (détecté à partir des sons normaux)
THRESHOLD_PATH = MODEL_DIR / "threshold.npy"

# 📄 Fichier CSV des prédictions reconstruction/anomalie
PREDICTIONS_FILE = PREDICTIONS_DIR / "reconstruction_predictions.csv"

# ==========================================
# 🎛️ Audio Feature Extraction Parameters
# ==========================================
RANDOM_STATE = 72           # Random State
TARGET_FRAMES = 128
SR = 16000                  # Sample rate (Hz)
N_MELS = 64                 # Number of Mel frequency bins
N_MFCC = 14                 # Number of MFCC coefficients
N_FFT = 512                 # Window size for FFT (2048)
HOP_LENGTH = 247            # Window size for FFT (2048)
SEGMENT_DURATION_SEC = 2    # Duration of audio segments
TOTAL_DURATION_SEC = 10     # Total audio duration

# 🎯 Liste des familles de features extraites (utile pour filtrage, sélection, affichage)
FEATURES_USED = [
    "mfcc_combined",       # MFCC + delta + delta²
    "mel_spec",            # Mel-spectrogram (log)
    "spectral_contrast",   # Contraste spectral (harmoniques)
    "rms"                  # Root Mean Square energy
]

feature_groups = {
    "contextual": [
        "file_id", "filename", "machine_type", "label",
        "n_samples", "duration_sec", "norm_method"
    ],

    "temporal_features": [
        "rms_mean", "rms_std", "zcr_mean", "zcr_std"
    ],

    "spectral_features": [
        "centroid_mean", "centroid_std",
        "rolloff_mean", "rolloff_std"
    ],

    "contrast_features": [
        "contrast_global_mean",
        *[f"spectral_contrast_{i}" for i in range(7)],
        *[f"contrast_std_{i}" for i in range(7)]
    ],

    "ber_features": [
        "ber_global_mean", "ber_0", "ber_1", "ber_2", "ber_3"
    ],

    "mfcc_features": [
        "mfcc_global_mean", "mfcc_global_std",
        *[f"mfcc_mean_{i}" for i in range(N_MFCC)],
        *[f"mfcc_std_{i}" for i in range(N_MFCC)]
    ],

    "delta_features": [
        "delta1_mfcc_global_mean", "delta1_mfcc_global_std",
        "delta2_mfcc_global_mean", "delta2_mfcc_global_std",
        *[f"delta1_mean_{i}" for i in range(N_MFCC)],
        *[f"delta2_mean_{i}" for i in range(N_MFCC)]
    ],
}

all_features = (
    feature_groups["temporal_features"]
    + feature_groups["spectral_features"]
    + feature_groups["contrast_features"]
    + feature_groups["ber_features"]
    + feature_groups["mfcc_features"]
    + feature_groups["delta_features"]
)

# ==========================================
# ✅ Ensure all necessary directories exist
# ==========================================
# ✅ Création automatique des dossiers manquants
for directory in [
    PROJECT_DATA_DIR,
    AUDIO_DATA_ROOT,
    DCASE_DATASET_DIR,
    DEV_AUDIO_DIR,
    NUMPY_DATA_DIR,
    FEATURES_DIR,
    REDUCED_DIR,
    MODEL_DIR,
    SCALER_DIR,
    IMAGES_DIR,
    PREDICTIONS_DIR,
    PROCESSED_DIR,
    LOG_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)

