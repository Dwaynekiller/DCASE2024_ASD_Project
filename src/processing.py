"""
processing.py

Utility functions for audio data processing, feature extraction, and dataset management.

Provides:
  - load_npy_files: Loads preprocessed `.npy` audio files.
  - extract_features: Extracts multiple types of features from audio data.
  - save_features: Saves extracted features in a structured format.
  - load_features_dataset: Loads saved feature matrices for training.

Author: [Your Name]
Date: 2025-03-12
"""
#import os
#import scipy
import time
import datetime
import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt
import IPython.display as ipd
import librosa.display
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, List, Optional

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
#from sklearn.manifold import TSNE
#from sklearn.metrics import confusion_matrix
#import itertools
#from mpl_toolkits.mplot3d import Axes3D
from src.utils.logger_utils import logger, handle_errors

VERBOSE_LOG = True  # ou True selon ton besoin

@handle_errors
def preprocess_audio_and_save(
    file_path: str,
    output_dir: str,
    target_duration: int = 10,
    segment_duration: int = 5
) -> Tuple[List[Tuple[str, int, float]], int]:
    """
    Load, normalize, and preprocess an audio file to a fixed duration (trim if longer),
    then segment it in chunks of `segment_duration` seconds, and save them as `.npy`.

    Args:
        file_path (str):
            Path to the input audio file (.wav).
        output_dir (str):
            Directory where the processed `.npy` files will be saved.
        target_duration (int, optional):
            Desired total duration of the audio in seconds (default: 10).
            If the audio is longer, we trim it. If shorter, raise an error.
        segment_duration (int, optional):
            Duration (in seconds) of each chunk to be saved separately (default: 5).

    Returns:
        Tuple[List[Tuple[str, int, float]], int]:
            - A list of tuples (output_path, sample_rate, segment_duration)
              for each segment saved.
            - The sample rate (sr).

    Raises:
        FileNotFoundError:
            If the input audio file does not exist.
        ValueError:
            If the audio is shorter than `target_duration`.
        Exception:
            Other exceptions are handled by the @handle_errors decorator.
    """
    file_path_obj = Path(file_path)
    if not file_path_obj.exists():
        raise FileNotFoundError(f"Audio file '{file_path_obj}' does not exist.")

    # 🔄 Load audio file
    y, sr = librosa.load(file_path_obj, sr=None, res_type='kaiser_fast')
    if VERBOSE_LOG:
        logger.info(f"Audio loaded: {file_path_obj.name} (sample_rate={sr})")

    target_length = int(target_duration * sr)
    current_length = len(y)
    current_sec = current_length / sr

    # Trim audio if longer than target_duration
    if current_sec > target_duration:
        y = y[:target_length]
        if VERBOSE_LOG:
            logger.info(f"Audio trimmed: '{file_path_obj.name}' " f"({current_sec:.2f}s → {target_duration}s)")
    elif current_sec < target_duration:
        msg = (
            f"Audio '{file_path_obj.name}' is shorter "
            f"({current_sec:.2f}s) than required ({target_duration}s)."
        )
        logger.error(msg)
        raise ValueError(msg)

    # Normalize audio signal
    y = librosa.util.normalize(y)

    # Ensure output directory exists
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 🕒 Découpage en segments de `segment_duration`
    seg_samples = int(segment_duration * sr)
    nb_segments = current_length // seg_samples

    processed_files = []

    for i in range(nb_segments):
        start_idx = i * seg_samples
        end_idx = (i + 1) * seg_samples
        segment = y[start_idx:end_idx]

        # (Optionnel) Réduction de bruit
        # segment = nr.reduce_noise(y=segment, sr=sr)

        # 📌 Stockage en .npy
        seg_path = out_dir / f"{file_path_obj.stem}_seg{i}.npy"
        np.save(seg_path, segment)
        
        if VERBOSE_LOG:
            logger.info(f"Répertoire  {i}: '{seg_path}' -> " f"Processed segment {i}: '{file_path_obj.name}' -> " f"'{seg_path.name}' (Duration: {segment_duration}s, SR={sr})")
        processed_files.append((str(seg_path), sr, float(segment_duration)))

    return processed_files, sr

@handle_errors
def load_audio_features(
    features_dir: str,
    flatten: bool = True,
    scaler: Optional[StandardScaler] = None
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load and combine 'mel_raw' + 'other_raw' features, returning X, y, and a metadata DataFrame.

    - If flatten=True => ML case:
        * Concat mel+other => (n_feat, frames)
        * format_features(..., flatten=True, scaler=scaler) => (n_feat*frames,)
    - If flatten=False => DL case:
        * mel => format_features(mel, flatten=False, scaler=MinMaxScaler())
        * other => format_features(other, flatten=False, scaler=StandardScaler())
        * Concat => (n_feat_mel+n_feat_other, frames)
        * Pas de scaler global sur l'ensemble

    Args:
        features_dir (str): Directory with *mel_raw.npy and *other_raw.npy.
        flatten (bool, optional): 
            If True => one single vector for ML approach.
            If False => keep 2D shape for a DL approach.
        scaler (Optional[StandardScaler], optional):
            A global scaler (ex: StandardScaler) to apply in flatten=True scenario.

    Returns:
        X (np.ndarray): shape (N, dim) or (N, n_features, frames)
        y (np.ndarray): shape (N,) with 0/1
        df_meta (pd.DataFrame): metadata

    Raises:
        FileNotFoundError: If features_dir does not exist.
        ValueError: If mismatch shapes or no data found.
    """
    features_path = Path(features_dir)
    if not features_path.exists():
        raise FileNotFoundError(f"Features directory '{features_path}' does not exist.")

    logger.info(f"🔄 load_audio_features from: {features_path}")
    start_time = time.time()

    X_list = []
    y_list = []
    meta_rows = []

    mel_files = sorted(features_path.rglob("*_mel_raw.npy"))
    logger.info(f"Found {len(mel_files)} mel_raw files in total.")

    for mel_file in tqdm(mel_files, desc="Loading mel+other", unit="file"):
        stem = mel_file.stem.replace("_mel_raw", "")
        other_file = mel_file.parent / f"{stem}_other_raw.npy"

        if not other_file.exists():
            logger.warning(f"Missing other_raw for '{mel_file.name}' => skip.")
            continue

        # Chargement
        mel_raw = np.load(mel_file)
        other_raw = np.load(other_file)

        # Vérif shape frames
        if mel_raw.shape[1] != other_raw.shape[1]:
            raise ValueError(
                f"Inconsistent frames: mel={mel_raw.shape}, other={other_raw.shape}, file '{mel_file.name}'"
            )

        # Label
        label = 1 if "anomaly" in mel_file.stem else 0
        y_list.append(label)

        # Récup info path
        parts = mel_file.parts
        if len(parts) >= 4:
            machine_type = parts[-4]
            data_split = parts[-3]
            machine_id = parts[-2]
        else:
            machine_type = "unknown"
            data_split = "unknown"
            machine_id = "unknown"

        meta_rows.append({
            "filename": mel_file.name,
            "machine_type": machine_type,
            "machine_id": machine_id,
            "data_split": data_split,
            "condition": "anomaly" if label == 1 else "normal",
            "mel_shape": mel_raw.shape,
            "other_shape": other_raw.shape
        })

        # ==== Distinction flatten ou pas ====
        if flatten:
            # ML case: on concat direct => shape (mel_dim+other_dim, frames)
            feats_concat = np.concatenate([mel_raw, other_raw], axis=0)
            # => (128, frames) par ex.
            # Flatten + scaler
            feats_final = format_features(
                features=feats_concat,
                flatten=True,        # aplatit => (128*frames,)
                scaler=scaler        # ex: StandardScaler
            )
            X_list.append(feats_final)

        else:
            # DL case: on normalise localement mel et other
            mel_norm = format_features(
                features=mel_raw,
                flatten=False,
                scaler=MinMaxScaler()
            )
            other_norm = format_features(
                features=other_raw,
                flatten=False,
                scaler=StandardScaler()
            )
            # Concat => (mel_dim + other_dim, frames)
            feats_concat = np.concatenate([mel_norm, other_norm], axis=0)
            # Pas de flatten final
            X_list.append(feats_concat)

    # Fin de boucle
    if not X_list:
        raise ValueError("No features loaded. Possibly missing pairs or mismatch shapes?")

    # Empilement final
    try:
        X = np.array(X_list, dtype=np.float32)
    except ValueError as ve:
        logger.error(f"Inhomogeneous shapes in X_list => {ve}")
        raise

    y = np.array(y_list, dtype=np.int8)
    df_meta = pd.DataFrame(meta_rows)

    logger.info(
        f"✅ Loaded {X.shape[0]} samples. X shape={X.shape}, y shape={y.shape}. "
        f"df_meta shape={df_meta.shape} "
        f"Time elapsed={datetime.timedelta(seconds=time.time()-start_time)}"
    )

    return X, y, df_meta

@handle_errors
def format_features(
    features: np.ndarray,
    flatten: bool = False,
    scaler: Optional[object] = None
) -> np.ndarray:
    """
    Format and normalize audio features for model training or inference.
    
    Args:
        features (np.ndarray): 
            Array of shape (n_features, frames).
        flatten (bool, optional): 
            If True, flatten to shape (n_features * frames,).
            If False, keep shape (n_features, frames).
        scaler (optional):
            An instance of a scaler (StandardScaler, MinMaxScaler, etc.) 
            applied column-wise (frames as columns).
            If None, no scaling is done.
            
    Returns:
        np.ndarray: A 2D or 1D array, depending on flatten.
        
    Raises:
        ValueError: If features array is empty.
    """
    if features.size == 0:
        raise ValueError("Input features array is empty for format_features!")

    if VERBOSE_LOG:
        logger.info(f"[format_features] Original shape: {features.shape}")

    # Scaler si fourni
    if scaler is not None:
        # On transpose => shape (frames, n_features)
        # On applique scaler => on retranspose
        feats_scaled = scaler.fit_transform(features.T).T
        if VERBOSE_LOG:
            logger.info(f"[format_features] Features scaled with {scaler.__class__.__name__}.")
    else:
        feats_scaled = features

    # Flatten si besoin
    if flatten:
        feats_final = feats_scaled.flatten()
        if VERBOSE_LOG:
            logger.info(f"[format_features] Flatten -> shape: {feats_final.shape}")
    else:
        feats_final = feats_scaled
        if VERBOSE_LOG:
            logger.info(f"[format_features] Keeping 2D shape: {feats_final.shape}")

    return feats_final

@handle_errors
def load_metadata(
    data_path: str,
    output_path: str,
    output_file: str,
    target_duration: int = 10,
    segment_duration: int = 5
) -> pd.DataFrame:
    """
    Creates (or loads if exists) a DataFrame with simplified metadata for audio files.

    It scans the 'data_path' directory to find .wav files following a certain naming
    convention (*id*.wav), extracts minimal info (machine, condition, etc.), and calls
    'preprocess_audio_and_save()' to generate .npy files trimmed/normalized at 'target_duration'.
    Then it saves a simplified metadata CSV file for future usage.

    Args:
        data_path (str): Path containing the original audio files (e.g., dev_data).
        output_path (str): Directory where processed .npy files and metadata CSV are saved.
        output_file (str): Name of the CSV metadata file (without full path).
        target_duration (int, optional): Desired duration of audio (seconds). Defaults to 10.
        segment_duration (int, optional): Desired crop of audio (seconds). Defaults to 5.

    Returns:
        pd.DataFrame: A DataFrame with minimal metadata:
            - file_path: str, path to the processed .npy file
            - machine_type: str (parent folder)
            - machine_id: str (extracted from filename)
            - sample_id: str
            - data_split: str (train/test, etc.)
            - condition: str (normal/anomaly)
            - duration: float16 (the target_duration)
            - sampling_rate: int16
            - filename: original .wav filename

    Raises:
        FileNotFoundError: If data_path does not exist.
        Exception: Other exceptions are logged and raised by the @handle_errors decorator.
    """
    data_path_obj = Path(data_path)
    if not data_path_obj.exists():
        raise FileNotFoundError(f"Data path '{data_path_obj}' does not exist.")

    metadata_path = Path(output_path) / output_file

    # If CSV already exists, load and return it
    if metadata_path.exists():
        logger.info(f"Loading existing metadata file: '{output_file}'")
        return pd.read_csv(
            metadata_path,
            dtype={
                "file_path": "str",
                "machine_type": "str",
                "machine_id": "str",
                "data_split": "str",
                "condition": "str",
                "duration": "float16",
                "sampling_rate": "int16"
            }
        )

    logger.info(f"Creating metadata file: '{output_file}'")
    start_time = time.time()

    # List relevant audio files
    audio_files = sorted(data_path_obj.glob("*/*/*/*id*.wav"))
    metadata = []

    for file_path in tqdm(audio_files, desc="Processing audio files", unit="file"):
        file_name = file_path.name
        try:
            parts = file_name.split("_")
            condition = parts[0]  # e.g. normal, anomaly
            machine_id = f"{parts[1]}_{parts[2]}"  # e.g. id_00
            sample_id = parts[3].replace(".wav", "")  # ex. '00000000'
            machine_type = file_path.parent.parent.parent.name
            data_split = file_path.parent.parent.name

            # Create output directory for processed .npy
            class_dir = Path(output_path) / machine_type / data_split / machine_id
            class_dir.mkdir(parents=True, exist_ok=True)
            if VERBOSE_LOG:
                logger.info(f"class_dir: '{class_dir}' ")

            # Preprocess audio file (normalize, trim to target_duration, save .npy)
            # Returns a list of (segment_path, sr, seg_duration) or equivalent
            processed, sr = preprocess_audio_and_save(
                file_path=file_path,
                output_dir=str(class_dir),
                target_duration=target_duration,
                segment_duration=segment_duration
            )

            # For each segment, add a row in metadata
            for seg_info in processed:
                seg_path, seg_sr, seg_dur = seg_info
                # Add essential metadata
                metadata.append({
                    "file_path": str(seg_path),
                    "filename": file_name,
                    "machine_type": machine_type,
                    "machine_id": machine_id,
                    "sample_id": sample_id,
                    "data_split": data_split,
                    "condition": condition,
                    "duration": float(seg_dur),
                    "sampling_rate": int(seg_sr)
                })

        except Exception as exc:
            logger.error(f"Error processing '{file_name}': {exc}")

    # Create DataFrame from metadata
    df = pd.DataFrame(metadata).astype({
        "file_path": "str",
        "machine_type": "str",
        "machine_id": "str",
        "data_split": "str",
        "condition": "str",
        "duration": "float16",
        "sampling_rate": "int16"
    })

    # Save simplified metadata to CSV
    df.to_csv(metadata_path, index=False)
    elapsed_time = time.time() - start_time
    logger.info(
        f"Metadata saved successfully to '{output_file}'. "
        f"Processing time: {str(datetime.timedelta(seconds=elapsed_time))}"
    )

    # Shuffle rows to avoid any order bias
    return df.sample(frac=1, random_state=6472).reset_index(drop=True)

@handle_errors
def extract_features(
    audio_data: np.ndarray,
    sr: int,
    hop_length: int,
    n_mels: int,
    n_fft: int,
    n_mfcc: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract a comprehensive set of audio features from a signal, returning two arrays:
      1. Mel spectrogram (in dB) -> shape (n_mels, frames)
      2. MFCC stack + other spectral features -> shape (?, frames)

    Features extracted:
        - Mel Spectrogram (dB) [mel_db]
        - MFCCs + delta + delta² [mfcc_stack]
        - Chroma, Spectral Contrast, Centroid, Rolloff, RMS [other]
        - Then we combine MFCC stack + other -> [mfcc_stack_other]

    Args:
        audio_data (np.ndarray):
            Audio waveform in mono.
        sr (int):
            Sampling rate.
        hop_length (int):
            Hop length for frame-based features.
        n_mels (int):
            Number of Mel bands for Mel spectrogram.
        n_fft (int):
            FFT window size.
        n_mfcc (int):
            Number of MFCC coefficients.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            (mel_db, mfcc_stack_other)
            - mel_db: shape = (n_mels, frames)
            - mfcc_stack_other: shape = (some_dim, frames)
    
    Raises:
        ValueError:
            If the audio data is empty.
    """
    if audio_data.size == 0:
        raise ValueError("Audio data is empty!")

    # 🔍 Mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio_data,
        sr=sr,
        n_mels=n_mels,
        hop_length=hop_length,
        n_fft=n_fft
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    if VERBOSE_LOG:
        logger.info(f"[extract_features] mel_db shape: {mel_db.shape}")

    # 🔍 MFCC
    mfcc = librosa.feature.mfcc(
        y=audio_data,
        sr=sr,
        n_mfcc=n_mfcc,
        hop_length=hop_length,
        n_fft=n_fft
    )
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)

    mfcc_stack = np.concatenate([mfcc, delta_mfcc, delta2_mfcc], axis=0)
    if VERBOSE_LOG:
        logger.info(f"[extract_features] mfcc_stack shape: {mfcc_stack.shape}")

    # 🔍 Other features
    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr, hop_length=hop_length, n_fft=n_fft)
    spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sr, hop_length=hop_length, n_fft=n_fft)
    spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr, hop_length=hop_length, n_fft=n_fft)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr, hop_length=hop_length, n_fft=n_fft)
    #zcr = librosa.feature.zero_crossing_rate(y=audio_data, hop_length=hop_length)
    rms = librosa.feature.rms(y=audio_data, frame_length=n_fft, hop_length=hop_length)

    other = np.concatenate([chroma, spectral_contrast, spectral_centroid, spectral_rolloff, rms], axis=0)
    if VERBOSE_LOG:
        logger.info(f"[extract_features] other shape: {other.shape}")

    # 🔗 Combine MFCC stack + other
    mfcc_stack_other = np.concatenate([mfcc_stack, other], axis=0)
    if VERBOSE_LOG:
        logger.info(f"[extract_features] combined shape: {mfcc_stack_other.shape}")

    # 📝 (Optionnel) Normalisation locale
    # from sklearn.preprocessing import StandardScaler, MinMaxScaler
    # mel_db = MinMaxScaler().fit_transform(mel_db.T).T
    # mfcc_stack_other = StandardScaler().fit_transform(mfcc_stack_other.T).T
    if VERBOSE_LOG:
            logger.info("[extract_features] Feature extraction done.")

    return mel_db, mfcc_stack_other

@handle_errors
def process_all_npy_files(
    numpy_dir: str,
    features_dir: str,
    sr: int,
    hop_length: int,
    n_mels: int,
    n_fft: int,
    n_mfcc: int,
    target_duration: int,
    segment_duration: int
) -> None:
    """
    Process all `.npy` files to extract features and save them.

    Args:
        numpy_dir (str): Directory containing `.npy` audio files (normalized).
        features_dir (str): Directory to save extracted features.
        sr (int): Sample rate.
        hop_length (int): Base hop_length (for 10s).
        n_mels (int): Number of Mel frequency bands.
        n_fft (int): FFT window size.
        n_mfcc (int): Number of MFCC coefficients.
        target_duration (int): Typical target duration (e.g., 10s).
        segment_duration (int): Typical segment duration (e.g., 5s) if 2 segments per file.

    Raises:
        FileNotFoundError: If the numpy_dir does not exist.
        Exception: If there is an error processing the files.
    """
    numpy_path = Path(numpy_dir)
    if not numpy_path.exists():
        raise FileNotFoundError(f"Numpy directory {numpy_path} does not exist!")

    features_path = Path(features_dir)
    features_path.mkdir(parents=True, exist_ok=True)

    logger.info("Starting feature extraction from .npy audio files...")
    start_time = time.time()

    # Process each .npy file
    # Suppose the structure is machine_type/data_split/machine_id/*.npy
    npy_files = sorted(numpy_path.glob("*/*/*/*.npy"))
    for npy_file in tqdm(npy_files, desc="Processing .npy files", unit="file"):
        try:
            # 1) Load audio data
            audio_data = np.load(npy_file)
            duration_sec = len(audio_data) / sr

            # 2) Determine correct hop_length
            # e.g. if near 10s => hop_length = the base
            # if near 5s => hop_length = half
            # else skip or handle differently
            if abs(duration_sec - target_duration) < 0.5:
                cur_hop = hop_length
            elif abs(duration_sec - segment_duration) < 0.5:
                cur_hop = hop_length // 2
            else:
                logger.warning(
                    f"File {npy_file.name} has an unexpected duration {duration_sec:.2f}s. Skipping."
                )
                continue

            # 3) Extract features
            # Suppose extract_features returns (mel_raw, other_raw)
            mel_raw, other_raw = extract_features(
                audio_data=audio_data,
                sr=sr,
                hop_length=cur_hop,
                n_mels=n_mels,
                n_fft=n_fft,
                n_mfcc=n_mfcc
            )

            # 4) Build the output path
            # e.g. machine_type/data_split/machine_id/
            machine_type, data_split, machine_id = npy_file.parts[-4], npy_file.parts[-3], npy_file.parts[-2]
            out_dir = features_path / machine_type / data_split / machine_id
            out_dir.mkdir(parents=True, exist_ok=True)

            # 5) Save features
            mel_file = out_dir / f"{npy_file.stem}_mel_raw.npy"
            other_file = out_dir / f"{npy_file.stem}_other_raw.npy"
            np.save(mel_file, mel_raw)
            np.save(other_file, other_raw)
            
            if VERBOSE_LOG:
                logger.info(f"Features saved: {mel_file.name}, {other_file.name}")

        except Exception as exc:
            logger.error(f"Error processing {npy_file}: {exc}")

    elapsed_time = time.time() - start_time
    logger.info(
        "Feature extraction completed successfully. "
        f"Processing time: {datetime.timedelta(seconds=elapsed_time)}."
    )

@handle_errors
def split_train_test(X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training and test sets.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Label vector.
        test_size (float, optional): Proportion of the dataset to include in the test split. Defaults to 0.2.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Training and test sets for features and labels.

    Raises:
        ValueError: If X and y have different lengths.
        Exception: If there is an error splitting the data.
    """
    if len(X) != len(y):
        raise ValueError("X and y must have the same length!")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    if VERBOSE_LOG:
        logger.info(f"Data split: {len(X_train)} train, {len(X_test)} test")
    return X_train, X_test, y_train, y_test
    

#@handle_errors
#def load_sound_file(audio_path: str, offset: float = 0.0, duration: float = None) -> tuple[np.ndarray, int, float]:
#    """
#    Load an audio file and return the signal, sample rate, and duration.
#
#    Args:
#        audio_path (str): Path to the audio file.
#        offset (float, optional): Offset in seconds before reading. Defaults to 0.0.
#        duration (float, optional): Duration to extract from the file. Defaults to None.
#
#    Returns:
#        tuple[np.ndarray, int, float]: Audio signal, sample rate, and duration.
#
#    Raises:
#        FileNotFoundError: If the audio file does not exist.
#        Exception: If there is an error loading the audio file.
#    """
#    file_path = Path(audio_path)
#    if not file_path.exists():
#        raise FileNotFoundError(f"Audio file {file_path} does not exist!")
#
#    # Load audio file
#    y, sr = librosa.load(file_path, sr=None, offset=offset, duration=duration, res_type='kaiser_fast')
#    duration = librosa.get_duration(y=y, sr=sr)
#    return y, sr, duration

#@handle_errors
#def rms_normalize_audio(y: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
#    """
#    Normalize audio signal based on RMS energy to a target RMS value.
#
#    Args:
#        y (np.ndarray): Audio signal array.
#        target_rms (float, optional): Desired RMS level. Default is 0.1.
#
#    Returns:
#        np.ndarray: RMS-normalized audio signal.
#
#    Raises:
#        ValueError: If the input signal is empty or has zero RMS.
#    """
#    if y.size == 0:
#        raise ValueError("Input audio signal is empty.")
#    
#    rms = np.sqrt(np.mean(y**2))
#    if rms == 0:
#        raise ValueError("RMS of input audio signal is zero.")
#
#    y_normalized = y * (target_rms / rms)
#
#    # Log RMS before and after normalization
#    #logger.info(f"RMS before: {rms:.6f}, after normalization: {np.sqrt(np.mean(y_normalized**2)):.6f}")
#
#    return y_normalized

#@handle_errors
#def load_features_dataset(features_dir: Path) -> tuple[np.ndarray, np.ndarray]:
#    """
#    Loads extracted feature datasets and creates labels.
#
#    Args:
#        features_dir (Path): Directory containing saved `.npy` feature files.
#
#    Returns:
#        tuple[np.ndarray, np.ndarray]: 
#            - X: Feature matrix.
#            - y: Labels (1 = anomaly, 0 = normal).
#    """
#    X, y = [], []
#    for machine_type in tqdm(features_dir.glob("*"), desc="🔄 Loading Machines"):
#        for split_dir in machine_type.glob("*"):
#            feature_files = list(split_dir.glob("*_features.npy"))
#
#            for file in feature_files:
#                features = np.load(file)
#                label = 1 if "anomaly" in file.name else 0
#
#                X.append(features)
#                y.append(label)
#
#    X = np.array(X, dtype=np.float16)
#    y = np.array(y, dtype=np.int8)
#
#    logger.info(f"✅ Loaded dataset: {X.shape[0]} samples, {X.shape[1]} features.")
#    return X, y

#@handle_errors
#def load_audio_features(
#    features_dir: str, 
#    flatten: bool = True,
#    scaler: Optional[StandardScaler] = None
#) -> tuple[np.ndarray, np.ndarray]:
#    """
#    Load preprocessed, format and normalize audio features and their corresponding labels for model training.
#
#    Args:
#        features_dir (str): Directory containing the feature files (e.g. other_raw and mel_raw).
#        flatten (bool, optional): 
#            If True, the features are flattened into a 1D vector.
#            If False, the original 2D shape is preserved.
#        scaler (Optional[StandardScaler], optional):
#            A scaler instance (e.g., StandardScaler) to normalize the features.
#            If None, no scaling is applied.
#
#    Returns:
#        tuple[np.ndarray, np.ndarray]: Feature matrix (X) and label vector (y).
#
#    Raises:
#        ValueError: If the input features are empty.
#    """
#    features_dir = Path(features_dir)
#    if not features_dir.exists():
#        raise FileNotFoundError(f"Features directory {features_dir} does not exist!")
#
#    start_time = time.time()
#    logger.info(f"Loading normalized features from {features_dir}...")
#
#    try:
#        X, X_Mel, X_Other, y = [], [], [], []
#        formatted_features = []
#        
#        # Iterate through machine types, splits and machine_id
#        for machine_type in tqdm(features_dir.glob("*"), desc="🔄 Processing machine types"):
#            if not machine_type.is_dir():
#                continue
#
#            # Iterate through split types ("train", "test")
#            for split_dir in tqdm(machine_type.glob("*"), desc=f"📂 {machine_type.name} - loading split files", leave=False):
#                if not split_dir.is_dir():
#                    continue
#
#                # Iterate through machine_id types ("id_*")
#                for id_dir in tqdm(split_dir.glob("*"), desc=f"📂 {split_dir.name} - loading machine_id files", leave=False):
#                    if not id_dir.is_dir():
#                        continue
#
#                    # Load feature files
#                    other_raw_files = list(id_dir.glob("*_other_raw.npy"))
#                    mel_raw_files = list(id_dir.glob("*_mel_raw.npy"))
#                    for mel_raw_file in mel_raw_files:
#                        try:
#                            # Load features and determine label
#                            features = np.load(mel_raw_file)
#                            label = 1 if "anomaly" in mel_raw_file.stem else 0  # 1 for anomaly, 0 for normal
#                            X_Mel.append(features)
#                            y.append(label)
#                            logger.info(f"Loaded X_Mel {features.shape}.")
#                        except Exception as e:
#                            logger.warning(f"⚠️ Error loading {mel_raw_file}: {e}")
#
#                    for other_raw_file in other_raw_files:
#                        try:
#                            # Load features and determine label
#                            features = np.load(other_raw_file)
#                            label = 1 if "anomaly" in other_raw_file.stem else 0  # 1 for anomaly, 0 for normal
#                            X_Other.append(features)
#                            y.append(label)
#                            logger.info(f"Loaded X_Other {features.shape}.")
#                        except Exception as e:
#                            logger.warning(f"⚠️ Error loading {other_raw_file}: {e}")                    
#
#        
#
#        if flatten:
#            formatted_features.append(X_Mel)
#            formatted_features.append(X_Other)
#            formatted_features = np.array(formatted_features, dtype=np.float16)  # ✅ Réduction mémoire
#            X = format_features(formatted_features, flatten=True, scaler=StandardScaler())
#            logger.info(f"Features flattened to shape: {X.shape}")
#        else:
#            X_Mel = np.array(X_Mel, dtype=np.float16)  # ✅ Réduction mémoire
#            formatted_features = format_features(X_Mel, flatten=False, scaler=MinMaxScaler())
#            X.append(formatted_features)
#            X_Other = np.array(X_Other, dtype=np.float16)  # ✅ Réduction mémoire
#            formatted_features = format_features(X_Other, flatten=False, scaler=StandardScaler())
#            X.append(formatted_features)
#        
#        elapsed_time = time.time() - start_time
#        logger.info(
#            f"Loaded {X.shape[0]} samples with {X.shape[1]} features each."
#            f"Label {y.shape[0]} samples with {y.shape[1]} features each."
#            f"Processing time: {str(datetime.timedelta(seconds=elapsed_time))}"
#        )
#
#        # Convert to numpy arrays
#        #X = np.array(X, dtype=np.float16)  # ✅ Réduction mémoire
#        y = np.array(y, dtype=np.int8)  # ✅ Encodage plus léger des labels (0 ou 1)
#
#        return X, y
#        
#    except Exception as e:
#        logger.error(f"❌ Erreur lors du chargement des features : {e}")
#        raise

