"""Patient-level train/val/test splitting for CHB-MIT.

Why patient-level? Windows from the same recording are highly correlated
(50% overlap, shared montage, same seizure). If windows from one patient
landed in both train and test, the model could memorise patient-specific
traits and the reported metrics would be optimistic. Splitting by *patient*
guarantees the val/test patients are never seen during feature selection
(Random Forest) or MLP training -- see feature-selection-strategy in memory.

The split is deterministic given a seed, so experiments are reproducible.
Downstream code filters per-recording feature files by the ``patient_id``
stored in each file's metadata against the split produced here.
"""

import json
import random
from pathlib import Path


def make_patient_split(patients, n_val=3, n_test=3, seed=42):
    """Assign patients to train/val/test sets, disjoint by patient.

    Args:
        patients: Iterable of patient IDs (e.g. ['chb01', ..., 'chb24']).
        n_val: Number of patients held out for validation.
        n_test: Number of patients held out for the final test set.
        seed: Random seed controlling the shuffle (reproducible).

    Returns:
        dict with keys 'train', 'val', 'test', each a sorted list of patient
        IDs. The three lists are disjoint and together cover all patients.

    Raises:
        ValueError: If there are not enough patients for the requested split
            or if duplicate patient IDs are supplied.
    """
    patients = list(patients)
    if len(set(patients)) != len(patients):
        raise ValueError("Duplicate patient IDs supplied to make_patient_split")
    if n_val + n_test >= len(patients):
        raise ValueError(
            f"Not enough patients ({len(patients)}) for n_val={n_val} + "
            f"n_test={n_test}; need at least one for training too"
        )

    # Deterministic shuffle. We use a local Random instance rather than the
    # global RNG so this never disturbs augmentation seeding elsewhere.
    rng = random.Random(seed)
    shuffled = sorted(patients)  # sort first so input order can't matter
    rng.shuffle(shuffled)

    test = shuffled[:n_test]
    val = shuffled[n_test:n_test + n_val]
    train = shuffled[n_test + n_val:]

    return {
        'train': sorted(train),
        'val': sorted(val),
        'test': sorted(test),
    }


def save_split(split, path, meta=None):
    """Write a split dict to JSON, with optional provenance metadata.

    Args:
        split: dict from make_patient_split.
        path: Output JSON path.
        meta: Optional dict of extra fields (seed, counts, etc.) recorded
            under a 'meta' key for traceability.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(split)
    if meta:
        payload['meta'] = meta
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)


def load_split(path):
    """Load a split JSON and return the {'train','val','test'} dict.

    The 'meta' key, if present, is ignored by callers that only need the
    patient assignments.
    """
    with open(path, 'r') as f:
        data = json.load(f)
    return {k: data[k] for k in ('train', 'val', 'test')}


def assign_split(patient_id, split):
    """Return which set ('train'/'val'/'test') a patient belongs to.

    Args:
        patient_id: e.g. 'chb07'.
        split: dict from make_patient_split / load_split.

    Returns:
        'train', 'val', or 'test'.

    Raises:
        KeyError: If the patient is not present in any set.
    """
    for name in ('train', 'val', 'test'):
        if patient_id in split[name]:
            return name
    raise KeyError(f"Patient {patient_id!r} not found in split")
