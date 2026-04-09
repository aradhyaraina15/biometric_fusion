# ─────────────────────────────────────────────
#  utils.py  —  shared helpers used by all modules
# ─────────────────────────────────────────────
import logging
import numpy as np

from .config import QUALITY_WINDOW


# ── Logger ────────────────────────────────────────────────────────────────────

def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s"
        ))
        logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger


# ── Image validation ──────────────────────────────────────────────────────────

def validate_image_shape(img: np.ndarray) -> bool:
    """Return True only if img is a non-empty 2-D array."""
    if img is None:
        return False
    if img.ndim != 2:
        return False
    if img.shape[0] == 0 or img.shape[1] == 0:
        return False
    return True


# ── Normalization ─────────────────────────────────────────────────────────────

def normalize_image(img: np.ndarray) -> np.ndarray:
    """Scale pixel values to [0.0, 1.0] as float32."""
    img = img.astype(np.float32)
    lo, hi = img.min(), img.max()
    if hi - lo == 0:
        return np.zeros_like(img)
    return (img - lo) / (hi - lo)


# ── Local quality score ───────────────────────────────────────────────────────

def compute_local_quality(img: np.ndarray, x: int, y: int) -> float:
    """
    Estimate ridge clarity around minutia (x, y) via local pixel std-dev.
    High std → good contrast → higher quality score.
    Returns float in [0.0, 1.0].
    """
    h, w = img.shape
    half = QUALITY_WINDOW // 2
    y1, y2 = max(0, y - half), min(h, y + half)
    x1, x2 = max(0, x - half), min(w, x + half)
    patch = img[y1:y2, x1:x2].astype(np.float32)
    if patch.size == 0:
        return 0.0
    return float(min(np.std(patch) / 64.0, 1.0))


# ── Circular angle difference ─────────────────────────────────────────────────

def angle_difference(a1: float, a2: float) -> float:
    """
    Smallest absolute difference between two angles in radians,
    correctly handling the 0 / 2π wrap-around.
    """
    diff = abs(a1 - a2) % (2 * np.pi)
    return float(min(diff, 2 * np.pi - diff))