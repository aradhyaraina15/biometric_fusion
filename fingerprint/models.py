# ─────────────────────────────────────────────
#  models.py  —  shared data structures
# ─────────────────────────────────────────────
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

class MinutiaeType(Enum):
    ENDING       = "ending"
    BIFURCATION  = "bifurcation"


@dataclass
class MinutiaePoint:
    x:       int
    y:       int
    angle:   float          # ridge orientation in radians
    type:    MinutiaeType
    quality: float = 1.0   # 0.0 (bad) to 1.0 (good)


@dataclass
class FingerprintResult:
    score:          float          # 0.0 – 1.0  (fusion-facing output)
    match:          bool           # score >= MATCH_THRESHOLD
    minutiae_count: int
    metadata:       dict = field(default_factory=dict)


@dataclass
class EmbeddingResult:
    """Output of the CNN encoder for one fingerprint image."""
    embedding:      np.ndarray   # 128-dimensional feature vector
    image_path:     str
    user_id:        str = ""