import re
from pathlib import Path

def parse_seizure_times(summary_file, edf_filename):
    """
    Parse seizure start/end times from CHB-MIT summary file.

    Args:
        summary_file: Path to *-summary.txt
        edf_filename: Name of the EDF file (e.g., 'chb01_03.edf')

    Returns:
        List of (start_sec, end_sec) tuples, empty list if no seizures
    """
    with open(summary_file, 'r') as f:
        content = f.read()

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


def check_window_seizure_label(window_start_sec, window_duration_sec, seizure_intervals, threshold=0.25):
    """
    Determine if a window should be labeled as seizure based on overlap.

    Args:
        window_start_sec: Window start time in seconds
        window_duration_sec: Window duration (e.g., 4 seconds)
        seizure_intervals: List of (start, end) tuples in seconds
        threshold: Minimum overlap ratio to label as seizure (default 0.25 = 25%)

    Returns:
        label: 1 if seizure, 0 if non-seizure
        overlap_ratio: Actual overlap ratio (0.0 to 1.0)
    """
    window_end_sec = window_start_sec + window_duration_sec

    total_overlap = 0.0
    for seizure_start, seizure_end in seizure_intervals:
        # Calculate overlap
        overlap_start = max(window_start_sec, seizure_start)
        overlap_end = min(window_end_sec, seizure_end)

        if overlap_start < overlap_end:
            total_overlap += (overlap_end - overlap_start)

    overlap_ratio = total_overlap / window_duration_sec
    label = 1 if overlap_ratio >= threshold else 0

    return label, overlap_ratio
