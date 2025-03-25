"""
config.py

Centralized configuration for audio anomaly detection (DCASE 2024 Task 2)

Defines:
- Directory structure (root, data, logs, models)
- Feature extraction parameters (FFT, MFCC, etc.)
- Paths for saving models, scalers, thresholds
"""
from pathlib import Path

## ==========================================
# 🧭 1. Localisation automatique du projet
# ==========================================

# Trouve automatiquement le dossier "Datascience" à partir du notebook courant
notebook_path = Path.cwd()
base_path = None
for parent in notebook_path.parents:
    if parent.name == "Datascience":
        base_path = parent
        break
if base_path is None:
    raise FileNotFoundError("Base directory 'Datascience' not found!")

# ==========================================
# 🏗️ 2. Répertoires de base du projet
# ==========================================
# 📁 Dossier racine du projet DCASE
ROOT_PROJECT_DIR = base_path / "Projet" / "Kaggle" / "DCASE2024_ASD_Project"

# 📁 Répertoire contenant les jeux de données audio bruts
ROOT_AUDIO_DIR = base_path / "Data" / "sound_datasets"

# 📁 Répertoire spécifique au dataset DCASE
DCASE_DIR = ROOT_AUDIO_DIR / "DCASE_DATASET"

# ==========================================
# 🏗️ 3. Processed audio and features
# ==========================================
# 📁 Fichiers audio convertis en .npy (après normalisation)
DEV_AUDIO_DIR = DCASE_DIR / "dev_data"

# 📁 Fichiers audio convertis en .npy (après normalisation)
NUMPY_AUDIO_DIR = DCASE_DIR / "numpy_data"

# 📁 Vecteurs de features extraits depuis les fichiers .npy
FEATURES_DIR = DCASE_DIR / "features"

# ==========================================
# 🏗️ 4. Model output and logs
# ==========================================
# 📁 Répertoire où seront enregistrés les modèles entraînés
MODEL_DIR = ROOT_PROJECT_DIR / "data" / "models"

# 📁 Répertoire pour les fichiers de prédictions / inférence
PREDICTIONS_DIR = ROOT_PROJECT_DIR / "data" / "outputs" / "predictions"

# 📁 Dossier pour les fichiers de logs
LOG_DIR = ROOT_PROJECT_DIR / "logs"

# ==========================================
# 🏗️ 5. Metadata
# ==========================================
# 📄 Fichier CSV des métadonnées (des fichiers .wav d'origine)
METADATA_FILE = DCASE_DIR / "dev_data.csv"

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

SR = 16000                # Sample rate (Hz)
N_MELS = 64               # Number of Mel frequency bins
N_MFCC = 14               # Number of MFCC coefficients
N_FFT = 512               # Window size for FFT (2048)
HOP_LENGTH = 466          # Window size for FFT (2048)
CROP_SEC = 5              #
DURATION_SEC = 10

# 🎯 Liste des familles de features extraites (utile pour filtrage, sélection, affichage)
FEATURES_USED = [
    "mfcc_combined",       # MFCC + delta + delta²
    "mel_spec",            # Mel-spectrogram (log)
    "spectral_contrast",   # Contraste spectral (harmoniques)
    "rms"                  # Root Mean Square energy
]


# ==========================================
# ✅ Ensure all necessary directories exist
# ==========================================
# ✅ Création automatique des dossiers manquants
for directory in [
    ROOT_PROJECT_DIR,
    ROOT_AUDIO_DIR,
    DCASE_DIR,
    NUMPY_AUDIO_DIR,
    DEV_AUDIO_DIR,
    FEATURES_DIR,
    MODEL_DIR,
    PREDICTIONS_DIR,
    LOG_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)

