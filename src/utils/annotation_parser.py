import re


def _read_summary(summary_file):
    # CHB-MIT summaries are ASCII.  utf-8 keeps that exact while making the
    # behavior explicit across Windows/Linux.
    with open(summary_file, 'r', encoding='utf-8') as handle:
        return handle.read()

def parse_seizure_times(summary_file, edf_filename):
    """
    Parse seizure start/end times from CHB-MIT summary file.

    Args:
        summary_file: Path to *-summary.txt
        edf_filename: Name of the EDF file (e.g., 'chb01_03.edf')

    Returns:
        List of (start_sec, end_sec) tuples, empty list if no seizures
    """
    content = _read_summary(summary_file)

    # Find the section for this file
    pattern = rf'File Name: {re.escape(edf_filename)}.*?(?=File Name:|$)'
    match = re.search(pattern, content, re.DOTALL)

    if not match:
        return []

    file_section = match.group(0)

    # Extract number of seizures
    num_seizures_match = re.search(r'Number of Seizures in File:\s*(\d+)', file_section)
    if not num_seizures_match or int(num_seizures_match.group(1)) == 0:
        return []

    # Extract seizure times - support both formats:
    # Format 1: "Seizure Start Time: 2996 seconds"
    # Format 2: "Seizure 1 Start Time: 2996 seconds"
    seizures = []

    # Try format with numbers first (Seizure 1, Seizure 2, etc.)
    numbered_starts = re.findall(r'Seizure \d+ Start Time:\s*(\d+)\s*seconds', file_section)
    numbered_ends = re.findall(r'Seizure \d+ End Time:\s*(\d+)\s*seconds', file_section)

    if numbered_starts and numbered_ends:
        # Use numbered format
        for start, end in zip(numbered_starts, numbered_ends):
            seizures.append((int(start), int(end)))
    else:
        # Fall back to unnumbered format
        start_times = re.findall(r'Seizure Start Time:\s*(\d+)\s*seconds', file_section)
        end_times = re.findall(r'Seizure End Time:\s*(\d+)\s*seconds', file_section)

        for start, end in zip(start_times, end_times):
            seizures.append((int(start), int(end)))

    return seizures


def is_seizure_time(time_sec, seizure_intervals):
    """Return True when ``time_sec`` lies in any [start, end) interval."""
    return any(start <= time_sec < end for start, end in seizure_intervals)


def label_window_center(window_start_sec, window_duration_sec, seizure_intervals):
    """Apply the locked 1-second-window label rule from README.md."""
    center_sec = window_start_sec + window_duration_sec / 2.0
    return int(is_seizure_time(center_sec, seizure_intervals))
