from dataclasses import asdict, dataclass
import json
from pathlib import Path


ALL_PATIENTS = tuple(f'chb{index:02d}' for index in range(1, 25))


@dataclass(frozen=True)
class DatasetSplit:
    name: str
    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]

    @classmethod
    def from_json(cls, path):
        with open(path, 'r', encoding='utf-8') as handle:
            payload = json.load(handle)
        split = cls(
            name=payload['name'],
            train=tuple(payload['train']),
            val=tuple(payload['val']),
            test=tuple(payload['test']),
        )
        split.validate()
        return split

    def validate(self):
        groups = {'train': self.train, 'val': self.val, 'test': self.test}
        flattened = [patient for values in groups.values() for patient in values]
        if len(flattened) != len(set(flattened)):
            raise ValueError('Patient split groups overlap.')
        unknown = sorted(set(flattened) - set(ALL_PATIENTS))
        missing = sorted(set(ALL_PATIENTS) - set(flattened))
        if unknown or missing:
            raise ValueError(
                f'Split must contain chb01..chb24 exactly once; '
                f'unknown={unknown}, missing={missing}'
            )

    def role_for(self, patient_id):
        for role in ('train', 'val', 'test'):
            if patient_id in getattr(self, role):
                return role
        raise KeyError(f'Patient {patient_id!r} is absent from split {self.name!r}.')

    def patients_for(self, roles):
        return tuple(
            patient
            for role in roles
            for patient in getattr(self, role)
        )


@dataclass(frozen=True)
class PreprocessingConfig:
    target_sfreq: float = 256.0
    highpass_hz: float = 0.1
    highpass_order: int = 2
    notch_hz: float = 60.0
    notch_q: float = 30.0
    rolling_std_sec: float = 600.0
    discard_initial_sec: float = 600.0
    scale: float = 0.2
    tanh_divisor: float = 1.2
    epsilon_volts: float = 1e-8
    window_sec: float = 1.0
    hop_sec: float = 0.5

    def __post_init__(self):
        if self.discard_initial_sec < self.rolling_std_sec:
            raise ValueError(
                'discard_initial_sec must be at least rolling_std_sec so every '
                'retained window uses a fully populated rolling history.'
            )

    def to_dict(self):
        return asdict(self)

    def validate_sfreq(self, sfreq):
        if abs(float(sfreq) - self.target_sfreq) > 1e-6:
            raise ValueError(
                f'Expected {self.target_sfreq:g} Hz, got {float(sfreq):g} Hz. '
                'Resampling is intentionally not implicit.'
            )

    def sample_counts(self):
        return {
            'rolling_std': int(round(self.rolling_std_sec * self.target_sfreq)),
            'discard_initial': int(round(
                self.discard_initial_sec * self.target_sfreq
            )),
            'window': int(round(self.window_sec * self.target_sfreq)),
            'hop': int(round(self.hop_sec * self.target_sfreq)),
        }


DEFAULT_SPLIT_CONFIG = (
    Path(__file__).resolve().parents[2]
    / 'configs'
    / 'split.json'
)
