import mne
import numpy as np
from pathlib import Path


class ChannelNotFoundError(ValueError):
    """Raised when a required target channel cannot be resolved in an EDF file.

    Carries the list of missing target channels so batch callers can report
    which files were skipped and why (e.g. chb12 referential/single-electrode
    recordings that have no bipolar T7-P7 / T8-P8 / FZ-CZ montage).
    """

    def __init__(self, missing, available):
        self.missing = missing
        self.available = available
        super().__init__(
            f"Missing target channel(s) {missing}. Available channels: {available}"
        )


# FZ-CZ substitutes, in priority order. Only used when FZ-CZ itself is absent.
# All 24 CHB-MIT patients contain FZ-CZ, so this is a safety net, not the
# normal path -- but keeping it deterministic avoids silent surprises.
FZ_CZ_ALTERNATIVES = ['CZ-PZ', 'C3-P3', 'C4-P4', 'F3-C3', 'F4-C4']


def _resolve_channel(target, available):
    """Resolve one target channel name to a concrete channel name in the file.

    Handles two real CHB-MIT quirks:
      1. Exact match (the common case).
      2. Duplicate channels: MNE renames repeated names such as two 'T8-P8'
         entries to 'T8-P8-0' / 'T8-P8-1'. We take the first occurrence, which
         is the primary channel in the standard 23-channel montage.

    Returns the resolved channel name, or None if the target is absent.
    """
    if target in available:
        return target

    # Duplicate-suffix match: 'T8-P8' -> 'T8-P8-0'. The remainder after the
    # target prefix must be '-<digits>' so we don't accidentally match an
    # unrelated montage like 'T8-P8X'.
    prefix = f"{target}-"
    duplicates = [
        ch for ch in available
        if ch.startswith(prefix) and ch[len(prefix):].isdigit()
    ]
    if duplicates:
        return duplicates[0]

    return None


def load_edf_channels(edf_path, target_channels, allow_fz_cz_alternative=True):
    """Load specific channels from an EDF file in the exact target order.

    Args:
        edf_path: Path to .edf file.
        target_channels: List of desired channel names, e.g.
            ['T7-P7', 'T8-P8', 'FZ-CZ']. The returned data rows are guaranteed
            to follow this order.
        allow_fz_cz_alternative: If True and 'FZ-CZ' is requested but absent,
            substitute the first available channel from FZ_CZ_ALTERNATIVES.

    Returns:
        data: numpy array [n_channels, n_samples], rows ordered to match
            target_channels.
        sfreq: Sampling frequency (Hz).
        final_channel_names: List of the actual EDF channel names used, in the
            same order as target_channels. For a substituted FZ-CZ this is the
            alternative's real name (e.g. 'CZ-PZ'); otherwise it equals the
            canonical target name.

    Raises:
        ChannelNotFoundError: If any target channel cannot be resolved. This
            lets batch processing skip incompatible files (e.g. chb12's CS2 and
            single-electrode recordings) instead of crashing the whole run.
    """
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    available_channels = raw.ch_names
    sfreq = raw.info['sfreq']

    resolved = []        # actual EDF channel names, in target order
    missing = []

    for target in target_channels:
        actual = _resolve_channel(target, available_channels)

        # FZ-CZ fallback: only when the canonical channel is truly absent.
        if actual is None and target == 'FZ-CZ' and allow_fz_cz_alternative:
            for alt in FZ_CZ_ALTERNATIVES:
                actual = _resolve_channel(alt, available_channels)
                if actual is not None:
                    print(f"  Note: 'FZ-CZ' absent, substituting '{actual}'")
                    break

        if actual is None:
            missing.append(target)
        else:
            resolved.append(actual)
            # Note duplicate-suffix resolutions ('T8-P8' -> 'T8-P8-0'). The
            # FZ-CZ substitution path already prints its own message above.
            if actual != target and actual.startswith(f"{target}-"):
                print(f"  Note: Using '{actual}' for '{target}' (duplicate channels found)")

    if missing:
        raise ChannelNotFoundError(missing, available_channels)

    # Pick the resolved channels, then reorder explicitly. MNE's pick_channels
    # keeps the file's original channel order, NOT the requested order, so we
    # must map back to target order to keep the downstream index contract
    # (index 2 == FZ-CZ in the feature extractor) correct.
    raw.pick_channels(resolved)
    data_all = raw.get_data()
    name_to_row = {name: i for i, name in enumerate(raw.ch_names)}
    data = np.stack([data_all[name_to_row[name]] for name in resolved])

    return data, sfreq, resolved


def find_fz_cz_alternative(available_channels):
    """Return FZ-CZ or the best available midline/central substitute.

    Retained for backward compatibility; load_edf_channels now performs this
    resolution internally via FZ_CZ_ALTERNATIVES.
    """
    for candidate in ['FZ-CZ'] + FZ_CZ_ALTERNATIVES:
        if candidate in available_channels:
            return candidate

    central_channels = [
        ch for ch in available_channels
        if 'C3' in ch or 'C4' in ch or 'CZ' in ch
    ]
    if central_channels:
        return central_channels[0]

    raise ChannelNotFoundError(['FZ-CZ or central alternative'], available_channels)
