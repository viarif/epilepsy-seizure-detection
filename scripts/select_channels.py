"""Select four canonical bipolar channels using training patients only."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.channel_selection import (
    audit_candidate_resolvability,
    save_selection,
    select_channels,
)
from src.preprocessing.config import ALL_PATIENTS, DEFAULT_SPLIT_CONFIG, DatasetSplit


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=Path('data/raw'))
    parser.add_argument('--split-config', type=Path, default=DEFAULT_SPLIT_CONFIG)
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/processed/channel_selection.json'),
    )
    parser.add_argument(
        '--patients',
        nargs='+',
        help='Optional TRAIN-patient subset for a smoke test.',
    )
    parser.add_argument(
        '--max-seizure-files-per-patient',
        type=int,
        default=None,
        help='Smoke-test limit; omit for the final selection.',
    )
    parser.add_argument(
        '--execute',
        action='store_true',
        help='Required to compute rankings. Without it, only header audit runs.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    split = DatasetSplit.from_json(args.split_config)
    train_patients = tuple(args.patients or split.train)
    invalid = sorted(set(train_patients) - set(split.train))
    if invalid:
        raise SystemExit(
            f'Channel selection may use train patients only; invalid={invalid}'
        )
    if not args.execute:
        audit = audit_candidate_resolvability(
            args.data_root,
            ALL_PATIENTS,
        )
        print(f"Header audit files: {audit['file_count']}")
        print(f"Usable candidates: {len(audit['usable_candidates'])}")
        print(f"Failures: {audit['failures']}")
        print('Ranking was NOT executed. Re-run with --execute after review.')
        return
    result = select_channels(
        args.data_root,
        train_patients=train_patients,
        availability_patients=ALL_PATIENTS,
        max_seizure_files_per_patient=args.max_seizure_files_per_patient,
    )
    # A result is final only when every configured training patient was used
    # and no smoke-test file limit was applied.
    result['complete'] = (
        train_patients == split.train
        and args.max_seizure_files_per_patient is None
    )
    result['split_name'] = split.name
    save_selection(result, args.output)
    print(f"Selected channels: {result['selected_channels']}")
    print(f"Complete final selection: {result['complete']}")
    print(f'Wrote: {args.output.resolve()}')


if __name__ == '__main__':
    main()
