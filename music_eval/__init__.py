"""Evaluation helpers for A/B-comparing model variants (see ../ROADMAP_V2.md).

Dependency-free (torch + torchaudio + scipy) fidelity and continuation metrics.
The fidelity metric (FMD) is a Fréchet *mel* distance — a no-embedding-model
proxy for FAD; swap in a learned embedding for true FAD when available.
"""
from .metrics import (
    mel_features,
    frechet_mel_distance,
    reference_mel_distance,
    seam_smoothness,
    style_consistency,
)

__all__ = [
    "mel_features",
    "frechet_mel_distance",
    "reference_mel_distance",
    "seam_smoothness",
    "style_consistency",
]
