from pathlib import Path

import mne
import numpy as np


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

# chb12_28/29 use ``01`` (zero-one) for the O1 electrode.  This is a known
# label typo in the downloaded CHB-MIT files, not a different electrode.
ELECTRODE_ALIASES = {
    'O1': ('01',),
}


def _target_aliases(target):
    """Return deterministic spelling variants for a canonical target."""
    aliases = [target]
    if target.count('-') == 1:
        left, right = target.split('-', 1)
        left_variants = (left,) + ELECTRODE_ALIASES.get(left, ())
        right_variants = (right,) + ELECTRODE_ALIASES.get(right, ())
        aliases = [
            f'{left_variant}-{right_variant}'
            for left_variant in left_variants
            for right_variant in right_variants
        ]
    else:
        aliases.extend(ELECTRODE_ALIASES.get(target, ()))
    return aliases


def _resolve_channel(target, available):
    """Resolve one target channel name to a concrete channel name in the file.

    Handles two real CHB-MIT quirks:
      1. Exact match (the common case).
      2. Duplicate channels: MNE renames repeated names such as two 'T8-P8'
         entries to 'T8-P8-0' / 'T8-P8-1'. We take the first occurrence, which
         is the primary channel in the standard 23-channel montage.

    Returns the resolved channel name, or None if the target is absent.
    """
    for alias in _target_aliases(target):
        if alias in available:
            return alias

        # Duplicate-suffix match: 'T8-P8' -> 'T8-P8-0'. The remainder after
        # the target prefix must be '-<digits>' so an unrelated montage cannot
        # be matched accidentally.
        prefix = f"{alias}-"
        duplicates = [
            ch for ch in available
            if ch.startswith(prefix) and ch[len(prefix):].isdigit()
        ]
        if duplicates:
            return sorted(duplicates)[0]

    return None


def _resolve_bipolar_derivation(target, available):
    """Resolve a bipolar target from compatible source channels.

    CHB-MIT is mostly stored as bipolar channels (for example ``T7-P7``),
    but chb12_27 uses a common ``CS2`` reference and chb12_28/29 store
    referential electrode channels.  The desired bipolar signal can be
    reconstructed exactly by subtraction::

        (T7-CS2) - (P7-CS2) == T7-P7
        T7 - P7                 == T7-P7

    A reversed bipolar channel is also usable after sign inversion.  Return a
    list of ``(channel_name, coefficient)`` source terms and a human-readable
    description, or ``(None, None)`` when no safe derivation is available.
    """
    if target.count('-') != 1:
        return None, None

    left, right = target.split('-', 1)

    # A reversed bipolar derivation only requires a sign flip.
    reversed_name = _resolve_channel(f'{right}-{left}', available)
    if reversed_name is not None:
        return [(reversed_name, -1.0)], f'-({reversed_name})'

    # Referential single-electrode channels with an implicit common reference.
    left_name = _resolve_channel(left, available)
    right_name = _resolve_channel(right, available)
    if left_name is not None and right_name is not None:
        return [(left_name, 1.0), (right_name, -1.0)], \
               f'{left_name} - {right_name}'

    # Explicit common-reference channels, e.g. T7-CS2 and P7-CS2.
    left_prefixes = [f'{name}-' for name in _target_aliases(left)]
    right_prefixes = [f'{name}-' for name in _target_aliases(right)]
    left_candidates = [
        ch for ch in available
        if any(ch.startswith(prefix) for prefix in left_prefixes)
    ]
    right_candidates = [
        ch for ch in available
        if any(ch.startswith(prefix) for prefix in right_prefixes)
    ]
    for left_source in left_candidates:
        left_prefix = next(
            prefix for prefix in left_prefixes if left_source.startswith(prefix)
        )
        left_reference = left_source[len(left_prefix):]
        for right_source in right_candidates:
            right_prefix = next(
                prefix for prefix in right_prefixes
                if right_source.startswith(prefix)
            )
            right_reference = right_source[len(right_prefix):]
            if left_reference == right_reference:
                return [(left_source, 1.0), (right_source, -1.0)], \
                       f'{left_source} - {right_source}'

    return None, None


def resolve_channel_sources(target, available_channels):
    """Resolve a canonical channel without reading signal samples.

    Returns:
        (source_terms, description), where ``source_terms`` is a list of
        ``(actual_edf_channel, coefficient)`` pairs.  ``source_terms`` is None
        when the target cannot be reconstructed safely.
    """
    actual = _resolve_channel(target, available_channels)
    if actual is not None:
        return [(actual, 1.0)], actual
    return _resolve_bipolar_derivation(target, available_channels)


def load_edf_channels(
    edf_path,
    target_channels,
    allow_fz_cz_alternative=False,
    start=None,
    stop=None,
    verbose=False,
):
    """Load specific channels from an EDF file in the exact target order.

    Args:
        edf_path: Path to .edf file.
        target_channels: List of desired channel names, e.g.
            ['T7-P7', 'T8-P8', 'FZ-CZ']. The returned data rows are guaranteed
            to follow this order.
        allow_fz_cz_alternative: If True and 'FZ-CZ' is requested but absent,
            substitute the first available channel from FZ_CZ_ALTERNATIVES.
            The preprocessing pipeline keeps this False because a selected
            canonical channel must not be changed in validation/test.
        start: Optional first sample to read (inclusive).
        stop: Optional last sample to read (exclusive).
        verbose: Print non-standard channel derivations when True.

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
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose='ERROR')
    available_channels = list(raw.ch_names)
    sfreq = float(raw.info['sfreq'])
    read_start = 0 if start is None else int(start)
    read_stop = raw.n_times if stop is None else int(stop)

    resolved = []        # source-term lists, in target order
    resolved_labels = [] # actual/derived channel descriptions, in target order
    missing = []

    for target in target_channels:
        source_terms, label = resolve_channel_sources(
            target, available_channels
        )
        actual = label
        if source_terms is not None and verbose and label != target:
            print(f"  Note: Resolving '{target}' as {label}")

        # FZ-CZ fallback: only when the canonical channel is truly absent.
        if source_terms is None and target == 'FZ-CZ' and allow_fz_cz_alternative:
            for alt in FZ_CZ_ALTERNATIVES:
                actual = _resolve_channel(alt, available_channels)
                if actual is not None:
                    source_terms = [(actual, 1.0)]
                    label = actual
                    if verbose:
                        print(
                            f"  Note: 'FZ-CZ' absent, substituting '{actual}'"
                        )
                    break

        if source_terms is None:
            missing.append(target)
        else:
            resolved.append(source_terms)
            resolved_labels.append(label)
            # Note duplicate-suffix resolutions ('T8-P8' -> 'T8-P8-0'). The
            # FZ-CZ substitution path already prints its own message above.
            if (len(source_terms) == 1 and source_terms[0][1] == 1.0
                    and label != target and label.startswith(f"{target}-")):
                if verbose:
                    print(
                        f"  Note: Using '{actual}' for '{target}' "
                        "(duplicate channels found)"
                    )

    if missing:
        raw.close()
        raise ChannelNotFoundError(missing, available_channels)

    # Read only the concrete EDF sources needed for the three outputs, then
    # apply subtraction/sign inversion.  This preserves the exact target order
    # and avoids copying every channel from a long recording.
    source_names = []
    for terms in resolved:
        for name, _coefficient in terms:
            if name not in source_names:
                source_names.append(name)
    try:
        source_data = raw.get_data(
            picks=source_names,
            start=read_start,
            stop=read_stop,
        )
        source_by_name = {
            name: source_data[i] for i, name in enumerate(source_names)
        }
        direct_sources = [
            terms[0][0]
            for terms in resolved
            if len(terms) == 1 and terms[0][1] == 1.0
        ]
        if (
            len(direct_sources) == len(resolved)
            and direct_sources == source_names
        ):
            # Standard CHB-MIT bipolar files already have the requested rows in
            # canonical order. Reuse MNE's array instead of making a second
            # full-size 18-channel copy.
            data = source_data
        else:
            data = np.stack([
                sum(
                    coefficient * source_by_name[name]
                    for name, coefficient in terms
                )
                for terms in resolved
            ])
    finally:
        raw.close()

    return data, sfreq, resolved_labels


def inspect_edf(edf_path):
    """Read EDF metadata only; no signal samples are preloaded."""
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose='ERROR')
    try:
        return {
            'path': str(Path(edf_path)),
            'channels': list(raw.ch_names),
            'sfreq': float(raw.info['sfreq']),
            'n_samples': int(raw.n_times),
        }
    finally:
        raw.close()


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
