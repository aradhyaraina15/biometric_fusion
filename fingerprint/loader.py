# ─────────────────────────────────────────────
#  loader.py  —  Stage 1: image input
# ─────────────────────────────────────────────
import cv2
import numpy as np

from .config import IMAGE_SIZE
from .utils  import setup_logger, validate_image_shape

logger = setup_logger(__name__)


class FingerprintLoader:

    def load_from_path(self, path: str) -> np.ndarray:
        """Read a fingerprint image file and return a normalised grayscale array."""
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image — check the path: {path}")
        if not validate_image_shape(img):
            raise ValueError(f"Image has invalid shape after loading: {img.shape}")
        logger.debug(f"Loaded '{path}'  shape={img.shape}")
        return self._to_standard_size(img)

    def load_from_array(self, array: np.ndarray) -> np.ndarray:
        """Accept a raw numpy array (e.g. from a live camera frame)."""
        if array is None:
            raise ValueError("Received None instead of an image array")
        if array.ndim == 3:
            array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
        if not validate_image_shape(array):
            raise ValueError(f"Array has invalid shape: {array.shape}")
        return self._to_standard_size(array)

    # ── private ───────────────────────────────────────────────────────────────

    def _to_standard_size(self, img: np.ndarray) -> np.ndarray:
        target_h, target_w = IMAGE_SIZE[1], IMAGE_SIZE[0]
        if img.shape[0] == target_h and img.shape[1] == target_w:
            return img
        resized = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
        logger.debug(f"Resized to {IMAGE_SIZE}")
        return resized