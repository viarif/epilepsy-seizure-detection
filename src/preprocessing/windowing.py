import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.eeg_loader import load_edf_channels, find_fz_cz_alternative
from src.utils.annotation_parser import parse_seizure_times, check_window_seizure_label


def create_sliding_windows(data, window_size, step_size):
    """
    Create sliding windows from continuous EEG data.

    Args:
        data: numpy array [n_channels, n_samples]
        window_size: Number of samples per window
        step_size: Step size in samples (overlap = window_size - step_size)

    Returns:
        windows: numpy array [n_windows, n_channels, window_size]
        window_starts: List of start indices for each window
    """
    n_channels, n_samples = data.shape

    # Calculate number of windows
    n_windows = (n_samples - window_size) // step_size + 1

    windows = []
    window_starts = []

    for i in range(n_windows):
        start_idx = i * step_size
        end_idx = start_idx + window_size

        if end_idx <= n_samples:
            window = data[:, start_idx:end_idx]
            windows.append(window)
            window_starts.append(start_idx)

    return np.array(windows), window_starts


def augment_seizure_window(window, n_augments=1, max_shift_samples=128):
    """
    Apply time-shift augmentation to seizure windows.

    Args:
        window: numpy array [n_channels, window_size]
        n_augments: Number of augmented copies to generate
        max_shift_samples: Maximum time shift in samples (±0.5s at 256Hz = ±128 samples)

    Returns:
        augmented: List of augmented windows
    """
    augmented = []
    n_channels, window_size = window.shape

    for _ in range(n_augments):
        # Random shift
        shift = np.random.randint(-max_shift_samples, max_shift_samples + 1)

        if shift == 0:
            augmented.append(window.copy())
        elif shift > 0:
            # Shift right, pad left with reflection
            shifted = np.zeros_like(window)
            shifted[:, shift:] = window[:, :-shift]
            shifted[:, :shift] = window[:, :shift][:, ::-1]  # Mirror padding
            augmented.append(shifted)
        else:
            # Shift left, pad right with reflection
            shift = abs(shift)
            shifted = np.zeros_like(window)
            shifted[:, :-shift] = window[:, shift:]
            shifted[:, -shift:] = window[:, -shift:][:, ::-1]  # Mirror padding
            augmented.append(shifted)

    return augmented


def process_edf_file(edf_path, summary_file, target_channels,
                     window_duration=4.0, overlap_ratio=0.5,
                     seizure_threshold=0.25, augment_seizures=True,
                     augment_overlap_threshold=0.5):
    """
    Process one EDF file: load, window, label, augment.

    Args:
        edf_path: Path to .edf file
        summary_file: Path to summary .txt file
        target_channels: List of desired channel names
        window_duration: Window size in seconds
        overlap_ratio: Overlap between windows (0.5 = 50%)
        seizure_threshold: Minimum overlap ratio to label as seizure (0.25 = 25%)
        augment_seizures: Whether to augment seizure windows
        augment_overlap_threshold: Only augment windows with overlap >= this (0.5 = 50%)

    Returns:
        X: numpy array [n_windows, n_channels, n_samples_per_window]
        y: numpy array [n_windows] with labels (0=normal, 1=seizure)
        info: dict with metadata
    """
    # Load EDF data
    print(f"Loading {Path(edf_path).name}...")
    data, sfreq, available_channels = load_edf_channels(edf_path, target_channels)

    # Check if we need to find alternative for FZ-CZ
    if 'FZ-CZ' in target_channels and 'FZ-CZ' not in available_channels:
        print(f"  FZ-CZ not found. Available channels: {available_channels}")
        alternative = find_fz_cz_alternative(available_channels)
        print(f"  Using alternative: {alternative}")

    print(f"  Loaded {len(available_channels)} channels: {available_channels}")
    print(f"  Sampling rate: {sfreq} Hz")
    print(f"  Duration: {data.shape[1] / sfreq:.1f} seconds")

    # Parse seizure annotations
    edf_filename = Path(edf_path).name
    seizure_intervals = parse_seizure_times(summary_file, edf_filename)
    print(f"  Seizure intervals: {seizure_intervals}")

    # Calculate windowing parameters
    window_samples = int(window_duration * sfreq)
    step_samples = int(window_samples * (1 - overlap_ratio))

    print(f"  Window size: {window_samples} samples ({window_duration}s)")
    print(f"  Step size: {step_samples} samples (overlap: {overlap_ratio*100:.0f}%)")

    # Create sliding windows
    windows, window_starts = create_sliding_windows(data, window_samples, step_samples)
    print(f"  Created {len(windows)} windows")

    # Generate labels and record overlap ratios
    labels = []
    overlap_ratios = []
    for start_idx in window_starts:
        start_sec = start_idx / sfreq
        label, overlap_ratio = check_window_seizure_label(start_sec, window_duration,
                                          seizure_intervals, seizure_threshold)
        labels.append(label)
        overlap_ratios.append(overlap_ratio)

    labels = np.array(labels)
    overlap_ratios = np.array(overlap_ratios)
    n_seizure = np.sum(labels == 1)
    n_normal = np.sum(labels == 0)
    print(f"  Labels: {n_seizure} seizure, {n_normal} normal (ratio 1:{n_normal/max(n_seizure,1):.1f})")

    # Apply data augmentation to seizure windows
    if augment_seizures and n_seizure > 0:
        print(f"  Applying augmentation to seizure windows (overlap >= {augment_overlap_threshold*100:.0f}%)...")
        X_list = []
        y_list = []

        n_augmented_windows = 0

        for window, label, overlap_ratio in zip(windows, labels, overlap_ratios):
            X_list.append(window)
            y_list.append(label)

            # Only augment windows with sufficient overlap to ensure label stability
            if label == 1 and overlap_ratio >= augment_overlap_threshold:
                # Generate 2 augmented copies
                augmented = augment_seizure_window(window, n_augments=2, max_shift_samples=128)
                X_list.extend(augmented)
                y_list.extend([1] * len(augmented))
                n_augmented_windows += 1

        X = np.array(X_list)
        y = np.array(y_list)

        n_seizure_aug = np.sum(y == 1)
        n_normal_aug = np.sum(y == 0)
        print(f"  Augmented {n_augmented_windows} windows (overlap >= {augment_overlap_threshold*100:.0f}%)")
        print(f"  After augmentation: {n_seizure_aug} seizure, {n_normal_aug} normal (ratio 1:{n_normal_aug/max(n_seizure_aug,1):.1f})")
    else:
        X = windows
        y = labels

    info = {
        'edf_file': edf_filename,
        'channels': available_channels,
        'sfreq': sfreq,
        'window_duration': window_duration,
        'overlap_ratio': overlap_ratio,
        'seizure_threshold': seizure_threshold,
        'augment_overlap_threshold': augment_overlap_threshold if augment_seizures else None,
        'n_windows': len(X),
        'n_seizure': np.sum(y == 1),
        'n_normal': np.sum(y == 0),
    }

    return X, y, info
