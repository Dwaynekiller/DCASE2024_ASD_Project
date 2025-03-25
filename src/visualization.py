# visualization.py

import numpy as np
#import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt
import librosa
import librosa.display
import scipy
from pathlib import Path
from tqdm import tqdm
#from sklearn.manifold import TSNE
#from sklearn.metrics import confusion_matrix
#import itertools
#from mpl_toolkits.mplot3d import Axes3D
import IPython.display as ipd

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.utils.logger_utils import handle_errors, logger

# Set seaborn style
sns.set_style("whitegrid")

@handle_errors
def display_comparison_fourier(normal_path: str, anomaly_path: str, sr: int, machine_type: str = None, figsize: tuple = (24, 6)):
    """
    Display Fourier transform (frequency distribution) comparison of normal vs anomaly audio signals.

    Args:
        normal_path (str): Path to normal audio (.npy).
        anomaly_path (str): Path to anomaly audio (.npy).
        sr (int): Sampling rate.
        machine_type (str, optional): Machine type for plot title.
        figsize (tuple, optional): Figure size.
    """
    normal_signal = np.load(normal_path)
    anomaly_signal = np.load(anomaly_path)

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    title_prefix = f"{machine_type.capitalize()} - " if machine_type else ""

    # Normal
    normal_fft = np.abs(librosa.stft(normal_signal))
    axes[0].plot(normal_fft.mean(axis=1), color='blue', alpha=0.7)
    axes[0].set_title(f"{title_prefix}Normal Fourier", fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Frequency bin')
    axes[0].set_ylabel('Amplitude')

    # Anomaly
    anomaly_fft = np.abs(librosa.stft(anomaly_signal))
    axes[1].plot(anomaly_fft.mean(axis=1), color='red', alpha=0.7)
    axes[1].set_title(f"{title_prefix}Anomaly Fourier", fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Frequency bin')
    axes[1].set_ylabel('Amplitude')

    # Difference
    diff_fft = anomaly_fft.mean(axis=1) - normal_fft.mean(axis=1)
    axes[2].plot(diff_fft, color='purple', alpha=0.7)
    axes[2].set_title(f"{title_prefix}Fourier Difference (Anomaly - Normal)", fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Frequency bin')
    axes[2].set_ylabel('Amplitude Difference')

    plt.tight_layout()
    plt.show()

@handle_errors
def display_comparison_psd(normal_path: str, anomaly_path: str, sr: int, machine_type: str = None, figsize: tuple = (24, 6)):
    """
    Display power spectral density (PSD) comparison of normal vs anomaly signals.

    Args:
        normal_path (str): Path to normal audio (.npy).
        anomaly_path (str): Path to anomaly audio (.npy).
        sr (int): Sampling rate.
        machine_type (str, optional): Machine type for plot title.
        figsize (tuple, optional): Figure size.
    """
    normal_signal = np.load(normal_path)
    anomaly_signal = np.load(anomaly_path)

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    title_prefix = f"{machine_type.capitalize()} - " if machine_type else ""

    f_norm, psd_norm = scipy.signal.welch(normal_signal, sr)
    f_anom, psd_anom = scipy.signal.welch(anomaly_signal, sr)

    # Normal PSD
    axes[0].semilogy(f_norm, psd_norm, color='blue')
    axes[0].set_title(f"{title_prefix}Normal PSD", fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Power')

    # Anomaly PSD
    axes[1].semilogy(f_anom, psd_anom, color='red')
    axes[1].set_title(f"{title_prefix}Anomaly PSD", fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Power')

    # Difference PSD
    psd_diff = psd_anom - psd_norm
    axes[2].plot(f_norm, psd_diff, color='purple')
    axes[2].set_title(f"{title_prefix}PSD Difference (Anomaly - Normal)", fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Frequency (Hz)')
    axes[2].set_ylabel('Power Difference')

    plt.tight_layout()
    plt.show()

@handle_errors
def display_comparison_melspectrogram(normal_path: str, anomaly_path: str, sr: int, machine_type: str = None, figsize: tuple = (24, 6)):
    """
    Display Mel Spectrogram comparison of normal vs anomaly signals.

    Args:
        normal_path (str): Path to normal audio (.npy).
        anomaly_path (str): Path to anomaly audio (.npy).
        sr (int): Sampling rate.
        machine_type (str, optional): Machine type for plot title.
        figsize (tuple, optional): Figure size.
    """
    normal_signal = np.load(normal_path)
    anomaly_signal = np.load(anomaly_path)

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    title_prefix = f"{machine_type.capitalize()} - " if machine_type else ""

    # Normal Mel
    mel_norm = librosa.power_to_db(librosa.feature.melspectrogram(y=normal_signal, sr=sr), ref=np.max)
    img1 = librosa.display.specshow(mel_norm, sr=sr, ax=axes[0], x_axis='time', y_axis='mel')
    axes[0].set_title(f"{title_prefix}Normal Mel-Spectrogram", fontsize=14, fontweight='bold')
    fig.colorbar(img1, ax=axes[0], format='%+2.0f dB')

    # Anomaly Mel
    mel_anom = librosa.power_to_db(librosa.feature.melspectrogram(y=anomaly_signal, sr=sr), ref=np.max)
    img2 = librosa.display.specshow(mel_anom, sr=sr, ax=axes[1], x_axis='time', y_axis='mel')
    axes[1].set_title(f"{title_prefix}Anomaly Mel-Spectrogram", fontsize=14, fontweight='bold')
    fig.colorbar(img2, ax=axes[1], format='%+2.0f dB')

    # Mel Difference
    mel_diff = mel_anom - mel_norm
    img3 = librosa.display.specshow(mel_diff, sr=sr, ax=axes[2], x_axis='time', y_axis='mel', cmap='coolwarm')
    axes[2].set_title(f"{title_prefix}Mel-Spectrogram Difference", fontsize=14, fontweight='bold')
    fig.colorbar(img3, ax=axes[2], format='%+2.0f dB')

    plt.tight_layout()
    plt.show()


@handle_errors
def inspect_npy_data(directory, sample_rate):
    """
    Inspect .npy audio files and construct a DataFrame with metadata including duration.

    Args:
        directory (Path): Directory containing .npy files structured by machine type and data split.
        sample_rate (int): Sample rate used for the audio files.

    Returns:
        pd.DataFrame: DataFrame containing audio metadata.
    
    Raises:
        ValueError: If sample_rate is invalid.
    """
    if sample_rate <= 0:
        raise ValueError("Sample rate must be a positive integer.")

    data_summary = []

    for machine_type in tqdm(sorted(directory.glob("*")), desc="Inspection des types de machines"):
        if machine_type.is_dir():
            for data_split in sorted(machine_type.glob("*")):
                npy_files = list(data_split.glob("*.npy"))
                for npy_file in npy_files:
                    condition = 'anomaly' if 'anomaly' in npy_file.stem else 'normal'

                    audio_length = np.load(npy_file).shape[0]
                    duration = audio_length / sample_rate  # 🔥 nom corrigé ici au singulier !

                    data_summary.append({
                        'machine_type': machine_type.name,
                        'data_split': data_split.name,
                        'condition': condition,
                        'file_path': npy_file,
                        'duration': duration  # 🔥 nom corrigé ici aussi
                    })

    df = pd.DataFrame(data_summary)
    logger.info(f"DataFrame created with {len(df)} audio files inspected.")
    return df

@handle_errors
def plot_audio_duration_distribution(
    dataframe: pd.DataFrame, bins: int = 30, figsize: tuple = (10, 6)
):
    """
    Plot a histogram and summary statistics of audio duration.

    Args:
        dataframe (pd.DataFrame): DataFrame containing 'duration' column.
        bins (int, optional): Number of bins for the histogram. Default is 30.
        figsize (tuple, optional): Figure size. Default is (10, 6).

    Raises:
        ValueError: If 'duration' column is missing.
        Exception: General exceptions handled by decorator (logs the error).
    """
    if 'duration' not in dataframe.columns:
        raise ValueError("The DataFrame must contain a 'duration' column.")

    duration = dataframe['duration']

    # Plot histogram
    plt.figure(figsize=figsize)
    sns.histplot(duration, bins=bins, kde=True, color='skyblue')
    plt.xlabel('Duration (seconds)', fontsize=12)
    plt.ylabel('Number of Audio Files', fontsize=12)
    plt.title('Distribution of Audio duration', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

    # Summary statistics clearly presented
    stats_df = duration.describe().to_frame().reset_index()
    stats_df.columns = ['Statistic', 'Duration (s)']
    stats_df['Duration (s)'] = stats_df['Duration (s)'].apply(lambda x: f"{x:.2f}")

    display(stats_df)

@handle_errors
def plot_catplot(dataframe: pd.DataFrame, data_split: str = None, figsize: tuple = (9, 5)):
    """
    Plot the distribution of audio files by machine type, condition, and data split.

    Args:
        dataframe (pd.DataFrame): DataFrame containing audio metadata.
        data_split (str, optional): Filter data by a specific subset (e.g., "train", "test"). Default is None.
        figsize (tuple, optional): Size of the displayed plots. Default is (9, 5).

    Raises:
        ValueError: If required columns are missing from the DataFrame.
        Exception: General exception handled by decorator (logs the error).
    """
    # Check required columns
    required_columns = {"machine_type", "condition", "data_split"}
    if not required_columns.issubset(dataframe.columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")

    # Filter DataFrame by data_split if specified
    df_filtered = dataframe.copy()
    if data_split is not None:
        df_filtered = df_filtered[df_filtered["data_split"] == data_split]

    # Plot 1: Distribution by machine type and condition
    plt.figure(figsize=figsize)
    sns.countplot(
        data=df_filtered,
        x="machine_type",
        hue="condition",
        palette="coolwarm"
    )
    plt.title("Distribution by Machine Type and Condition", fontsize=14, fontweight="bold")
    plt.xlabel("Machine Type", fontsize=12)
    plt.ylabel("Number of Files", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(title="Condition", title_fontsize="11", fontsize="10", loc="best")
    plt.tight_layout()
    plt.show()

    # Plot 2: Distribution by machine type and data_split (if multiple splits exist)
    if df_filtered["data_split"].nunique() > 1:
        plt.figure(figsize=figsize)
        sns.countplot(
            data=df_filtered,
            x="machine_type",
            hue="data_split",
            palette="viridis"
        )
        plt.title("Distribution by Machine Type and Data Split", fontsize=14, fontweight="bold")
        plt.xlabel("Machine Type", fontsize=12)
        plt.ylabel("Number of Files", fontsize=12)
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(title="Data Split", title_fontsize="11", fontsize="10", loc="best")
        plt.tight_layout()
        plt.show()

import IPython.display as ipd

@handle_errors
def display_comparison_waveforms(
    normal_path: str, anomaly_path: str, sr: int, machine_type: str = None, figsize: tuple = (24, 6)
):
    """
    Display waveforms of two audio files (normal, anomaly, and overlap) and provide audio playback.

    Args:
        normal_path (str): Path to the normal audio `.npy` file.
        anomaly_path (str): Path to the anomalous audio `.npy` file.
        sr (int): Sample rate of the audio signals.
        machine_type (str, optional): Machine type for title. Defaults to None.
        figsize (tuple, optional): Figure size. Defaults to (24, 6).

    Raises:
        FileNotFoundError: If either audio file does not exist.
        Exception: General exceptions handled by decorator (logs the error).
    """
    normal_file = Path(normal_path)
    anomaly_file = Path(anomaly_path)

    if not normal_file.exists():
        raise FileNotFoundError(f"Normal audio file '{normal_file}' not found.")
    if not anomaly_file.exists():
        raise FileNotFoundError(f"Anomaly audio file '{anomaly_file}' not found.")

    # Load audio data
    normal_signal = np.load(normal_file)
    anomaly_signal = np.load(anomaly_file)

    # Plotting setup (3 plots: Normal, Anomaly, Overlap)
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    title_prefix = f"{machine_type.capitalize()} - " if machine_type else ""

    # Normal waveform
    librosa.display.waveshow(normal_signal, sr=sr, alpha=0.5, color="blue", linewidth=0.5, ax=axes[0])
    axes[0].set_title(f"{title_prefix}Normal Waveform", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")

    # Anomaly waveform
    librosa.display.waveshow(anomaly_signal, sr=sr, alpha=0.5, color="red", linewidth=0.5, ax=axes[1])
    axes[1].set_title(f"{title_prefix}Anomalous Waveform", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Amplitude")

    # Overlap waveform (Normal vs Anomaly)
    librosa.display.waveshow(normal_signal, sr=sr, alpha=0.5, color="blue", label="Normal", ax=axes[2])
    librosa.display.waveshow(anomaly_signal, sr=sr, alpha=0.5, color="red", label="Anomaly", ax=axes[2])
    axes[2].set_title(f"{title_prefix}Overlap: Normal vs Anomaly", fontsize=14, fontweight='bold')
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Amplitude")
    axes[2].legend(loc='best')

    plt.tight_layout()
    plt.show()

    # 📢 Audio playback
    print(f"🔊 Listen to '{machine_type.capitalize()}' - Normal Audio: {normal_file.name}")
    display(ipd.Audio(data=normal_signal, rate=sr))

    print(f"🔊 Listen to '{machine_type.capitalize()}' - Anomaly Audio: {anomaly_file.name}")
    display(ipd.Audio(data=anomaly_signal, rate=sr))

@handle_errors
def display_comparison_powerspectralforms(
    normal_path: str, anomaly_path: str, sr: int, machine_type: str = None, figsize: tuple = (24, 6)
):
    """
    Display power spectrogram comparisons between normal and anomalous audio signals.

    Args:
        normal_path (str): Path to the normal audio `.npy` file.
        anomaly_path (str): Path to the anomalous audio `.npy` file.
        sr (int): Sample rate of the audio signals.
        machine_type (str, optional): Type of machine for plot title. Defaults to None.
        figsize (tuple, optional): Figure size. Defaults to (24, 6).

    Raises:
        FileNotFoundError: If either audio file does not exist.
        Exception: General exceptions handled by decorator (logs the error).
    """
    normal_file = Path(normal_path)
    anomaly_file = Path(anomaly_path)

    if not normal_file.exists():
        raise FileNotFoundError(f"Normal audio file '{normal_file}' not found.")
    if not anomaly_file.exists():
        raise FileNotFoundError(f"Anomaly audio file '{anomaly_file}' not found.")

    # Load npy audio signals
    normal_signal = np.load(normal_file)
    anomaly_signal = np.load(anomaly_file)

    # Compute spectrograms
    normal_spec = librosa.amplitude_to_db(np.abs(librosa.stft(normal_signal)), ref=np.max)
    anomaly_spec = librosa.amplitude_to_db(np.abs(librosa.stft(anomaly_signal)), ref=np.max)

    # Plotting setup
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    title_prefix = f"{machine_type} - " if machine_type else ""

    # Normal signal spectrogram
    img1 = librosa.display.specshow(normal_spec, sr=sr, x_axis='time', y_axis='log', ax=axes[0])
    axes[0].set_title(f"{title_prefix}Normal Spectrogram", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Frequency (Hz)")
    fig.colorbar(img1, ax=axes[0], format="%+2.0f dB")

    # Anomaly signal spectrogram
    img2 = librosa.display.specshow(anomaly_spec, sr=sr, x_axis='time', y_axis='log', ax=axes[1])
    axes[1].set_title(f"{title_prefix}Anomaly Spectrogram", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Frequency (Hz)")
    fig.colorbar(img2, ax=axes[1], format="%+2.0f dB")

    # Overlapping Spectrograms (Difference)
    diff_spec = anomaly_spec - normal_spec
    img3 = librosa.display.specshow(diff_spec, sr=sr, x_axis='time', y_axis='log', cmap='coolwarm', ax=axes[2])
    axes[2].set_title(f"{title_prefix}Spectrogram Difference (Anomaly - Normal)", fontsize=14, fontweight='bold')
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Frequency (Hz)")
    fig.colorbar(img3, ax=axes[2], format="%+2.0f dB")

    plt.tight_layout()
    plt.show()

        
@handle_errors
def display_spectral_analysis(
    audio_path: str, 
    sr: int, 
    analysis_type: str = "spectrogram", 
    title: str = None, 
    size: tuple = (12, 8)
):
    """
    Display spectral analysis (spectrogram, mel spectrogram, or power spectral density) of audio.

    Args:
        audio_path (str): Path to `.npy` audio file.
        sr (int): Sample rate of the audio signal.
        analysis_type (str, optional): 'spectrogram', 'mel_spectrogram', or 'power_spectral_density'.
        title (str, optional): Plot title. Defaults to None.
        size (tuple, optional): Figure size. Defaults to (12, 8).

    Raises:
        FileNotFoundError: If audio file does not exist.
        ValueError: If an invalid analysis type is provided.
        Exception: General exceptions handled by decorator (logs the error).
    """
    file_path = Path(audio_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file {file_path} does not exist.")

    # Load npy file directly
    y = np.load(file_path)

    # Set default title if not provided
    if title is None:
        title = analysis_type.replace("_", " ").title()

    plt.figure(figsize=size)

    if analysis_type == "spectrogram":
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="log")
        plt.colorbar(format="%+2.0f dB")
        plt.ylabel("Frequency (Hz)")
        
    elif analysis_type == "mel_spectrogram":
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel")
        plt.colorbar(format="%+2.0f dB")
        plt.ylabel("Mel Frequency")
        
    elif analysis_type == "power_spectral_density":
        f, Pxx_den = scipy.signal.welch(y, sr, average="mean")
        plt.semilogy(f, Pxx_den)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("PSD (Power)")
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        
    else:
        raise ValueError(
            "Invalid analysis_type. Choose 'spectrogram', 'mel_spectrogram', or 'power_spectral_density'."
        )

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.show()

@handle_errors
def list_files(directory: Path) -> None:
    """
    Display the structure of normalized files in a directory.

    Args:
        directory (Path): Directory containing normalized files.

    Raises:
        FileNotFoundError: If the directory does not exist.
    """
    if not directory.exists():
        raise FileNotFoundError(f"Directory {directory} does not exist!")

    logger.info(f"✔ Content of directory {directory}:\n")
    for machine_type in sorted(directory.glob("*")):
        if not machine_type.is_dir():
            continue
        logger.info(f"✔ {machine_type.name}/")
        for split in sorted(machine_type.glob("*")):
            if not split.is_dir():
                continue
            logger.info(f"✔   {split.name}/")
            feature_files = list(split.glob("*.npy"))
            for file in feature_files:
                logger.info(f"✔     {file.name}")


@handle_errors
def display_waveform(audio_path: str, title: str = None, color: str = None, size: tuple = (14, 5)) -> None:
    """
    Display the waveform of an audio file.

    Args:
        audio_path (str): Path to the audio file.
        title (str, optional): Title of the plot. Defaults to the file name.
        color (str, optional): Color of the waveform. Defaults to None.
        size (tuple, optional): Figure size. Defaults to (14, 5).

    Raises:
        FileNotFoundError: If the audio file does not exist.
        Exception: If there is an error loading or processing the audio file.
    """
    try:
        file_path = Path(audio_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file {file_path} does not exist!")

        # Load audio file
        y, sr = librosa.load(file_path, sr=None)

        # Set default title if not provided
        if title is None:
            title = file_path.name

        # Plot waveform
        plt.figure(figsize=size)
        librosa.display.waveshow(y, sr=sr, color=color, alpha=0.5, linewidth=0.5)
        plt.title(title)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.show()

    except Exception as e:
        logger.error(f"❌ Error displaying waveform for {audio_path}: {e}")
        raise

@handle_errors
def tsne_scatter(features: np.ndarray, labels: np.ndarray, dimensions: int = 2) -> None:
    """
    Display a t-SNE scatter plot of audio features.

    Args:
        features (np.ndarray): Audio features.
        labels (np.ndarray): Labels for the features.
        dimensions (int, optional): Number of dimensions for t-SNE (2 or 3). Defaults to 2.

    Raises:
        ValueError: If dimensions is not 2 or 3.
        Exception: If there is an error generating the t-SNE plot.
    """
    try:
        if dimensions not in (2, 3):
            raise ValueError("t-SNE can only plot in 2D or 3D. Ensure 'dimensions' is 2 or 3.")

        # Perform t-SNE dimensionality reduction
        features_embedded = TSNE(n_components=dimensions, random_state=42).fit_transform(features)

        # Initialize the plot
        fig, ax = plt.subplots(figsize=(10, 10))

        if dimensions == 3:
            ax = fig.add_subplot(111, projection="3d")

        # Plot data
        for label, color in zip(np.unique(labels), ["brown", "orange", "violet", "blue", "red", "green"]):
            ax.scatter(
                *zip(*features_embedded[labels == label]),
                marker="o",
                color=color,
                s=3,
                alpha=0.7,
                label=label,
            )

        plt.title("t-SNE Result: Audio", weight="bold").set_fontsize("14")
        plt.xlabel("Dimension 1", weight="bold").set_fontsize("10")
        plt.ylabel("Dimension 2", weight="bold").set_fontsize("10")
        plt.legend(loc="best")
        plt.show()

    except Exception as e:
        logger.error(f"❌ Error generating t-SNE plot: {e}")
        raise


@handle_errors
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_label: list = None,
    title: str = None,
    cmap=None,
    size: tuple = (8, 6),
) -> None:
    """
    Display a confusion matrix for evaluating classification model performance.

    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
        class_label (list, optional): List of class labels. Defaults to None.
        title (str, optional): Title of the plot. Defaults to None.
        cmap (str, optional): Colormap for the plot. Defaults to 'Blues'.
        size (tuple, optional): Figure size. Defaults to (8, 6).

    Raises:
        ValueError: If y_true and y_pred have different lengths.
        Exception: If there is an error generating the confusion matrix.
    """
    try:
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length!")

        if cmap is None:
            cmap = plt.get_cmap("Blues")

        # Compute confusion matrix
        cm = confusion_matrix(y_true=y_true, y_pred=y_pred)

        accuracy = np.trace(cm) / np.sum(cm).astype("float")
        misclass = 1 - accuracy

        # Plot confusion matrix
        plt.figure(figsize=size)
        plt.imshow(cm, interpolation="nearest", cmap=cmap)
        plt.title(title or "Confusion Matrix")
        plt.colorbar()

        if class_label is not None:
            tick_marks = np.arange(len(class_label))
            plt.xticks(tick_marks, class_label, rotation=45)
            plt.yticks(tick_marks, class_label)

        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(
                j,
                i,
                cm[i, j],
                horizontalalignment="center",
                color="white" if cm[i, j] > (cm.max() / 2.0) else "black",
            )

        plt.tight_layout()
        plt.ylabel("True label")
        plt.xlabel(f"Predicted label\n\naccuracy={accuracy:.4f}; misclass={misclass:.4f}")
        plt.show()

    except Exception as e:
        logger.error(f"❌ Error generating confusion matrix: {e}")
        raise