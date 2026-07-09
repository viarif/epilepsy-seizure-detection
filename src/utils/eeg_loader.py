import mne
import numpy as np
from pathlib import Path

def load_edf_channels(edf_path, target_channels):
    """
    Load specific channels from EDF file.

    Args:
        edf_path: Path to .edf file
        target_channels: List of channel names to extract

    Returns:
        data: numpy array [n_channels, n_samples]
        sfreq: Sampling frequency
        available_channels: List of available channel names
    """
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    available_channels = raw.ch_names
    sfreq = raw.info['sfreq']

    # Handle duplicate channel names (e.g., T8-P8 -> T8-P8-0, T8-P8-1)
    channels_to_pick = []
    final_channel_names = []

    for target_ch in target_channels:
        if target_ch in available_channels:
            # Exact match found
            channels_to_pick.append(target_ch)
            final_channel_names.append(target_ch)
        else:
            # Check for duplicates with suffix (e.g., T8-P8-0, T8-P8-1)
            duplicates = [ch for ch in available_channels if ch.startswith(f"{target_ch}-")]
            if duplicates:
                # Use the first duplicate (T8-P8-0)
                channels_to_pick.append(duplicates[0])
                final_channel_names.append(target_ch)  # Use original name
                print(f"  Note: Using '{duplicates[0]}' for '{target_ch}' (duplicate channels found)")

    if len(channels_to_pick) == 0:
        raise ValueError(f"None of target channels {target_channels} found in {available_channels}")

    raw.pick_channels(channels_to_pick)
    data = raw.get_data()

    return data, sfreq, final_channel_names


def find_fz_cz_alternative(available_channels):
    """
    Find FZ-CZ channel or suitable alternative for midline reference.

    CHB-MIT uses bipolar montage. Common alternatives:
    - FZ-CZ (ideal midline)
    - C3-P3 or C4-P4 (central-parietal, often used)
    - F3-C3 or F4-C4 (frontal-central)

    Args:
        available_channels: List of channel names

    Returns:
        channel_name: Best available midline/central channel
    """
    # Priority list
    candidates = [
        'FZ-CZ',
        'CZ-PZ',
        'FZ-CZ',
        'C3-P3',  # Left central-parietal
        'C4-P4',  # Right central-parietal (symmetric alternative)
        'F3-C3',
        'F4-C4',
    ]

    for candidate in candidates:
        if candidate in available_channels:
            return candidate

    # Fallback: return any central channel
    central_channels = [ch for ch in available_channels if 'C3' in ch or 'C4' in ch or 'CZ' in ch]
    if central_channels:
        return central_channels[0]

    raise ValueError(f"No suitable midline/central channel found in {available_channels}")
