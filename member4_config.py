"""Detector settings; independent of the team's shared config.py."""
from dataclasses import dataclass


@dataclass(frozen=True)
class DetectorConfig:
    sample_rate: int = 16000
    hop: int = 256
    window_seconds: float = 0.5
    overlap: float = 0.5
    min_valid_fraction: float = 0.5
    benign_quantile: float = 0.01
    truth_overlap_fraction: float = 0.5

    def __post_init__(self):
        if self.sample_rate <= 0 or self.hop <= 0 or self.window_seconds <= 0:
            raise ValueError('Sample rate, hop and window must be positive')
        if not 0 <= self.overlap < 1:
            raise ValueError('overlap must be in [0, 1)')
        if not 0 < self.min_valid_fraction <= 1:
            raise ValueError('min_valid_fraction must be in (0, 1]')
        if not 0 < self.benign_quantile < 1:
            raise ValueError('benign_quantile must be in (0, 1)')
        if not 0 < self.truth_overlap_fraction <= 1:
            raise ValueError('truth_overlap_fraction must be in (0, 1]')
