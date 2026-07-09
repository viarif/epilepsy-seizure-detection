import numpy as np
from pathlib import Path
import re
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.eeg_loader import load_edf_channels
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


def augment_seizure_window(window, n_augments=1, max_shift_samples=128, random_seed=None):
    """
    Apply time-shift augmentation to seizure windows.

    Args:
        window: numpy array [n_channels, window_size]
        n_augments: Number of augmented copies to generate
        max_shift_samples: Maximum time shift in samples (±0.5s at 256Hz = ±128 samples)
        random_seed: Random seed for reproducibility (optional)

    Returns:
        augmented: List of augmented windows
        shifts: List of shift values applied (for metadata tracking)
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    augmented = []
    shifts = []
    n_channels, window_size = window.shape

    for _ in range(n_augments):
        # Random shift
        shift = np.random.randint(-max_shift_samples, max_shift_samples + 1)
        shifts.append(shift)

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

    return augmented, shifts


def process_edf_file(edf_path, summary_file, target_channels,
                     window_duration=4.0, overlap_ratio=0.5,
                     seizure_threshold=0.25, augment_seizures=True,
                     augment_overlap_threshold=0.5, random_seed=42,
                     patient_id=None):
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
        random_seed: Random seed for augmentation reproducibility
        patient_id: Patient ID for the split contract (e.g. 'chb17'). If None,
            it is inferred from the filename prefix -- but that is WRONG for
            multi-part recordings like 'chb17a_03.edf' (which would yield
            'chb17a'). Callers that know the source directory should pass the
            directory name so patient-level splitting stays correct.

    Returns:
        X: numpy array [n_windows, n_channels, n_samples_per_window]
        y: numpy array [n_windows] with labels (0=normal, 1=seizure)
        metadata: dict with detailed metadata for each window
    """
    # Set random seed for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)
    # Load EDF data. The loader resolves duplicate-suffix names and any FZ-CZ
    # substitute internally, and returns rows in target_channels order so the
    # downstream feature extractor's channel-index contract stays valid.
    print(f"Loading {Path(edf_path).name}...")
    data, sfreq, actual_channels = load_edf_channels(edf_path, target_channels)

    print(f"  Loaded {len(actual_channels)} channels: {actual_channels}")
    print(f"  Sampling rate: {sfreq} Hz")
    print(f"  Duration: {data.shape[1] / sfreq:.1f} seconds")

    # Parse seizure annotations
    edf_filename = Path(edf_path).name
    if patient_id is None:
        # Fallback inference. Strip a trailing letter from the numeric-prefix
        # part so 'chb17a_03.edf' -> 'chb17', not 'chb17a'. Callers should pass
        # patient_id explicitly (the directory name) to avoid relying on this.
        prefix = edf_filename.split('_')[0]
        patient_id = re.sub(r'[a-z]$', '', prefix)
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

    # Generate labels and record per-window seizure-overlap fractions.
    # NOTE: use a distinct name (`seizure_overlap`) for the window/seizure
    # overlap so we never shadow the `overlap_ratio` *parameter* (the sliding
    # step overlap). They mean different things and mixing them silently
    # corrupted the saved metadata before.
    labels = []
    seizure_overlaps = []
    window_metadata = []

    for window_idx, start_idx in enumerate(window_starts):
        start_sec = start_idx / sfreq
        label, seizure_overlap = check_window_seizure_label(start_sec, window_duration,
                                          seizure_intervals, seizure_threshold)
        labels.append(label)
        seizure_overlaps.append(seizure_overlap)

        # Store metadata for each window. `seizure_overlap` is the fraction of
        # this window covered by a labelled seizure (0.0 for background files).
        window_metadata.append({
            'window_idx': window_idx,
            'start_sample': start_idx,
            'start_time_sec': start_sec,
            'end_time_sec': start_sec + window_duration,
            'seizure_overlap': seizure_overlap,
            'is_augmented': False,
            'augment_shift': None,
            'original_window_idx': window_idx
        })

    labels = np.array(labels)
    seizure_overlaps = np.array(seizure_overlaps)
    n_seizure = np.sum(labels == 1)
    n_normal = np.sum(labels == 0)
    print(f"  Labels: {n_seizure} seizure, {n_normal} normal (ratio 1:{n_normal/max(n_seizure,1):.1f})")

    # Apply data augmentation to seizure windows
    if augment_seizures and n_seizure > 0:
        print(f"  Applying augmentation to seizure windows (overlap >= {augment_overlap_threshold*100:.0f}%)...")
        X_list = []
        y_list = []
        metadata_list = []

        n_augmented_windows = 0
        augment_counter = 0

        for window, label, seizure_overlap, meta in zip(windows, labels, seizure_overlaps, window_metadata):
            X_list.append(window)
            y_list.append(label)
            metadata_list.append(meta)

            # Only augment windows with sufficient overlap to ensure label stability
            if label == 1 and seizure_overlap >= augment_overlap_threshold:
                # Generate 2 augmented copies with unique seeds
                base_seed = random_seed + augment_counter if random_seed is not None else None
                augmented, shifts = augment_seizure_window(window, n_augments=2,
                                                          max_shift_samples=128,
                                                          random_seed=base_seed)

                # Add augmented windows with metadata
                for aug_idx, (aug_window, shift) in enumerate(zip(augmented, shifts)):
                    X_list.append(aug_window)
                    y_list.append(1)

                    # Create metadata for augmented window
                    aug_meta = meta.copy()
                    aug_meta['is_augmented'] = True
                    aug_meta['augment_shift'] = shift
                    aug_meta['augment_idx'] = aug_idx
                    aug_meta['window_idx'] = len(metadata_list)  # Update to new index
                    metadata_list.append(aug_meta)

                n_augmented_windows += 1
                augment_counter += 1

        X = np.array(X_list)
        y = np.array(y_list)

        n_seizure_aug = np.sum(y == 1)
        n_normal_aug = np.sum(y == 0)
        print(f"  Augmented {n_augmented_windows} windows (overlap >= {augment_overlap_threshold*100:.0f}%)")
        print(f"  After augmentation: {n_seizure_aug} seizure, {n_normal_aug} normal (ratio 1:{n_normal_aug/max(n_seizure_aug,1):.1f})")
    else:
        X = windows
        y = labels
        metadata_list = window_metadata

    metadata = {
        'patient_id': patient_id,
        'edf_file': edf_filename,
        'channels': actual_channels,
        'sfreq': sfreq,
        'window_duration': window_duration,
        'overlap_ratio': overlap_ratio,
        'seizure_threshold': seizure_threshold,
        'augment_overlap_threshold': augment_overlap_threshold if augment_seizures else None,
        'random_seed': random_seed,
        'n_windows_total': len(X),
        'n_windows_original': len(windows),
        'n_seizure': np.sum(y == 1),
        'n_normal': np.sum(y == 0),
        'seizure_intervals': seizure_intervals,
        'window_metadata': metadata_list  # Detailed per-window metadata
    }

    return X, y, metadata
