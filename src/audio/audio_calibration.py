"""
audio_calibration.py - Audio calibration utilities using centralized logging.
"""

import os
import numpy as np
import pandas as pd
import librosa
import math
import logging
from typing import Dict, List, Union, Optional, Tuple
from pathlib import Path

from src import config
from src.audio.audio_analysis_tools import load_sound_file
from src.utils.logger_utils import handle_errors, logger

# Initialisation du logger
logger = logging.getLogger("DCASE_PROJECT")

# Gestion robuste de l'import GPU
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = None
    GPU_AVAILABLE = False

@handle_errors
def get_best_hop_length_simple(
    duration: float,
    sr: int,
    n_fft: int,
    target_frames: int,
    center: bool = False,
    verbose: bool = False,
    exact_only: bool = True,
    selection_mode: str = 'median'
) -> int:
    """
    Compute the optimal hop_length for a given audio duration and target number of frames.

    Args:
        duration: Duration of the audio in seconds.
        sr: Sampling rate.
        n_fft: FFT window size (preferably a power of two).
        target_frames: Desired number of time frames.
        center: Whether to pad the signal in librosa (default: False).
        verbose: If True, log debug information (default: False).
        exact_only: If True, requires exact frame match (default: True).
        selection_mode: How to select from valid hops ('median', 'min', 'max') (default: 'median').

    Returns:
        Optimal hop_length that results in the target number of frames.
    """
    if duration <= 0 or sr <= 0 or n_fft <= 0 or target_frames <= 0:
        raise ValueError("All numerical parameters must be positive")

    if not (n_fft & (n_fft - 1) == 0) and verbose:
        logger.warning(f"Warning: n_fft={n_fft} is not a power of two")

    n_frames = int(sr * duration)
    if n_frames < n_fft:
        raise ValueError(f"Duration too short ({duration}s) for n_fft={n_fft}")
    
    N_eff = n_frames + (n_fft if center else 0)
    
    if verbose:
        logger.info(f'Parameters: N_eff={N_eff}, n_fft={n_fft}, target_frames={target_frames}')

    if target_frames == 1:
        if verbose:
            logger.info("Target frames is 1, returning n_fft as hop_length")
        return n_fft

    max_possible_frames = 1 + (N_eff - n_fft) // 1
    if target_frames > max_possible_frames:
        raise ValueError(f"Target frames {target_frames} impossible for duration {duration}s")

    max_hop = max(1, (N_eff - n_fft) // max(1, (target_frames - 1)))
    hop_range = np.arange(1, max_hop + 1, dtype=np.int32)
    frames = 1 + (N_eff - n_fft) // hop_range
    
    valid_hops = hop_range[frames == target_frames]
    
    if verbose:
        logger.info(f'Found {len(valid_hops)} valid hop lengths: {valid_hops}')

    if len(valid_hops) > 0:
        if selection_mode == 'median':
            hop_length = int(np.median(valid_hops))
        elif selection_mode == 'min':
            hop_length = int(valid_hops.min())
        elif selection_mode == 'max':
            hop_length = int(valid_hops.max())
        else:
            raise ValueError(f"Invalid selection_mode: {selection_mode}")
    elif not exact_only:
        hop_length = max(1, int(round((N_eff - n_fft) / (target_frames - 1))))
        if verbose:
            logger.info(f"No exact solution, using approximate hop_length: {hop_length}")
    else:
        raise ValueError(f"No valid hop_length found for duration={duration}s")

    mfcc = librosa.feature.mfcc(y=np.zeros(n_frames, dtype=np.float32), sr=sr, n_fft=n_fft, hop_length=hop_length, center=center)
    actual_frames = mfcc.shape[1]
    
    if exact_only and actual_frames != target_frames:
        raise ValueError(f"Verification failed: hop_length={hop_length} gives {actual_frames} frames")

    if verbose:
        logger.info(f"Final result: hop_length={hop_length} → {actual_frames} frames")

    return hop_length
    
@handle_errors
def is_reasonably_close(obtained: int, target: int, tolerance_pct: float = 2.0) -> bool:
    """
    Check if obtained value is reasonably close to target value.
    
    Args:
        obtained: Obtained value.
        target: Target value.
        tolerance_pct: Allowed percentage difference (default: 2.0).
    
    Returns:
        True if within tolerance, False otherwise.
    """
    if target <= 0 or obtained is None:
        return False
    error_pct = abs((obtained - target) / target * 100)
    return error_pct <= tolerance_pct

@handle_errors
def get_effective_length(n_frames: int, n_fft: int, center: bool = False) -> int:
    """
    Calculate effective length considering center padding.
    
    Args:
        n_frames: Number of audio frames.
        n_fft: FFT window size.
        center: Whether to center pad (default: False).
    
    Returns:
        Effective length.
    """
    return n_frames + (n_fft if center else 0)

@handle_errors
def compute_frames(N_eff: int, hop_length: int, n_fft: int) -> int:
    """
    Compute number of frames from effective length.
    
    Args:
        N_eff: Effective length.
        hop_length: Hop length.
        n_fft: FFT window size.
    
    Returns:
        Number of frames.
    """
    return 1 + (N_eff - n_fft) // hop_length

@handle_errors
def compute_valid_hop_lengths(
    n_frames: int,
    n_fft: int,
    target_frames: int,
    center: bool = False,
    device: str = 'cpu'
) -> Union[np.ndarray, 'cp.ndarray' if GPU_AVAILABLE else np.ndarray]:
    """
    Compute valid hop lengths for given parameters.
    
    Args:
        n_frames: Number of audio frames.
        n_fft: FFT window size.
        target_frames: Desired number of frames.
        center: Whether to center pad (default: False).
        device: Computation device ('cpu' or 'gpu') (default: 'cpu').
    
    Returns:
        Array of valid hop lengths.
    """
    if target_frames < 1:
        raise ValueError("target_frames must be ≥ 1")
    
    N_effective = get_effective_length(n_frames, n_fft, center)
    max_hop = max(1, (N_effective - n_fft) // max(1, (target_frames - 1)))
    
    if device == 'gpu' and GPU_AVAILABLE:
        try:
            hop_range = cp.arange(1, max_hop + 1, dtype=cp.int32)
            frames = compute_frames(N_eff=N_effective, hop_length=hop_range, n_fft=n_fft)
            valid_hops = hop_range[frames == target_frames].get()
            return valid_hops
        except Exception as e:
            print(f"GPU error, fallback to CPU: {str(e)}")
            device = 'cpu'
    
    hop_range = np.arange(1, max_hop + 1, dtype=np.int32)
    frames = compute_frames(N_eff=N_effective, hop_length=hop_range, n_fft=n_fft)
    return hop_range[frames == target_frames]

@handle_errors
def get_best_hop_length(
    duration: float,
    sr: int,
    n_fft: int,
    target_frames: int,
    center: bool = False,
    exact_only: bool = True
) -> dict:
    """
    Find best hop length with detailed results.
    
    Args:
        duration: Audio duration in seconds.
        sr: Sampling rate.
        n_fft: FFT window size.
        target_frames: Desired number of frames.
        center: Whether to center pad (default: False).
        exact_only: Require exact match (default: True).
    
    Returns:
        Dictionary with results.
    """
    n_frames = int(sr * duration)
    N_eff = get_effective_length(n_frames, n_fft, center)

    valid_hops = compute_valid_hop_lengths(
        n_frames=n_frames,
        n_fft=n_fft,
        target_frames=target_frames,
        center=center,
        device='cpu'
    )

    if len(valid_hops) > 0:
        hop_length = int(np.median(valid_hops))
        exact_match = True
    elif not exact_only:
        hop_length = int((N_eff - n_fft) / (target_frames - 1))
        exact_match = False
    else:
        raise ValueError(f"No valid hop_length found for duration={duration}s")

    mfcc = librosa.feature.mfcc(y=np.zeros(n_frames), sr=sr, n_fft=n_fft,
                                hop_length=hop_length, center=center)

    error_pct = round(100 * (mfcc.shape[1] - target_frames) / target_frames, 2)

    return {
        "hop_length": hop_length,
        "frames_obtained": mfcc.shape[1],
        "error_%": error_pct,
        "exact_match": mfcc.shape[1] == target_frames
    }

@handle_errors
def build_hop_table(
    durations_sec: List[float],
    target_frames_list: List[int],
    sr: int,
    n_fft: int,
    n_mfcc: int,
    n_mels: int,
    center: bool = False,
    exact_only: bool = True,
    include_shapes: bool = False
) -> pd.DataFrame:
    """
    Build table of hop lengths for various durations and target frames.
    
    Args:
        durations_sec: List of durations in seconds.
        target_frames_list: List of target frame counts.
        sr: Sampling rate.
        n_fft: FFT window size.
        n_mfcc: Number of MFCC coefficients (optional).
        n_mels: Number of Mel bands (optional).
        center: Whether to center pad (default: False).
        exact_only: Require exact match (default: True).
        include_shapes: Include feature shapes in output (default: False).
    
    Returns:
        DataFrame with results.
    """
    rows = []

    for dur in durations_sec:
        n_frames = int(sr * dur)
        for target in target_frames_list:
            row = {
                "duration": dur,
                "target_frames": target,
                "hop_length": None,
                "frames_obtained": None,
                "error_%": None,
                "exact_match": False,
                "MFCC_shape": None,
                "MEL_shape": None,
                "message": None
            }
            
            try:
                res = get_best_hop_length(
                    duration=dur,
                    sr=sr,
                    n_fft=n_fft,
                    target_frames=target,
                    center=center,
                    exact_only=exact_only
                )
                if res is None:
                    raise ValueError("get_best_hop_length returned None")

                row.update({
                    "hop_length": res["hop_length"],
                    "frames_obtained": res["frames_obtained"],
                    "error_%": res["error_%"],
                    "exact_match": res["exact_match"]
                })

                if include_shapes:
                    y = np.zeros(n_frames)
                    mfcc = librosa.feature.mfcc(
                        y=y, sr=sr, n_fft=n_fft,
                        hop_length=res["hop_length"],
                        n_mfcc=n_mfcc, center=center
                    )
                    mel_db = librosa.power_to_db(
                        librosa.feature.melspectrogram(
                            y=y, sr=sr, n_fft=n_fft,
                            hop_length=res["hop_length"],
                            n_mels=n_mels, center=center
                        ),
                        ref=np.max
                    )
                    row["MFCC_shape"] = mfcc.shape
                    row["MEL_shape"] = mel_db.shape

                rows.append(row)

            except Exception as e:
                row["message"] = str(e)
                rows.append(row)

    return pd.DataFrame(rows)

@handle_errors
def split_audio_into_segments(
    path: Path,
    output_dir: Path,
    mono: bool = True,
    duration: int = None,
    segment_duration: float = 2.0
) -> List[Path]:
    """
    Split audio file into fixed-duration segments.
    
    Args:
        path: Path to input audio file.
        output_dir: Directory to save segments.
        sr: Sampling rate.
        segment_duration: Duration of each segment in seconds (default: 2.0).
    
    Returns:
        List of paths to saved segments.
    """
    import soundfile as sf

    if not path.exists():
        logger.error(f"Audio file not found: {path}")
        raise FileNotFoundError(path)

    y = load_sound_file(path=path, sr=sr, duration=duration, mono=mono)
        
    total_samples = len(y)
    segment_samples = int(segment_duration * sr)
    total_segments = total_samples // segment_samples

    if total_segments == 0:
        raise ValueError("Audio is too short to split into segments.")

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    for i in range(total_segments):
        start = i * segment_samples
        end = start + segment_samples
        segment = y[start:end]

        segment_name = f"segment_{int(start/sr)}s_{int(end/sr)}s.wav"
        segment_path = output_dir / segment_name

        try:
            sf.write(segment_path, segment, sr)
            saved_paths.append(segment_path)
        except Exception as e:
            logger.error(f"Failed to save segment {i+1}: {e}")

    return saved_paths