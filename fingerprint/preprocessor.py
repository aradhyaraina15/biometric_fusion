# ─────────────────────────────────────────────
#  preprocessor.py  —  Stage 2: enhancement
# ─────────────────────────────────────────────
import cv2
import numpy as np
from skimage.morphology import skeletonize

from .config import CLAHE_CLIP_LIMIT, CLAHE_TILE_SIZE, GABOR_FREQUENCIES
from .utils  import setup_logger

logger = setup_logger(__name__)


class FingerprintPreprocessor:

    def process(self, image: np.ndarray) -> np.ndarray:
        """Full preprocessing pipeline: enhance → binarize → skeletonize."""
        enhanced = self._enhance(image)
        binary   = self._binarize(enhanced)
        skeleton = self._skeletonize(binary)
        logger.debug(f"Skeleton has {int(skeleton.sum())} ridge pixels")
        return skeleton

    # ── steps ─────────────────────────────────────────────────────────────────

    def _enhance(self, img: np.ndarray) -> np.ndarray:
        # 1. CLAHE — boost local contrast so ridges stand out
        clahe = cv2.createCLAHE(
            clipLimit=CLAHE_CLIP_LIMIT,
            tileGridSize=CLAHE_TILE_SIZE
        )
        img = clahe.apply(img)

        # 2. Gabor filter bank — amplify ridges, suppress noise
        accumulated = np.zeros_like(img, dtype=np.float32)
        for freq in GABOR_FREQUENCIES:
            for theta in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
                kernel = cv2.getGaborKernel(
                    ksize=(21, 21),
                    sigma=4.0,
                    theta=theta,
                    lambd=1.0 / freq,
                    gamma=0.5,
                    psi=0,
                    ktype=cv2.CV_32F
                )
                filtered = cv2.filter2D(img, cv2.CV_32F, kernel)
                accumulated += np.abs(filtered)

        # Normalise to uint8
        cv2.normalize(accumulated, accumulated, 0, 255, cv2.NORM_MINMAX)
        return accumulated.astype(np.uint8)

    def _binarize(self, img: np.ndarray) -> np.ndarray:
        # Otsu automatically finds the best threshold
        _, binary = cv2.threshold(
            img, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        # Optional: morphological closing to fill small gaps in ridges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        return binary

    def _skeletonize(self, binary: np.ndarray) -> np.ndarray:
        # skimage.skeletonize needs a boolean array
        skeleton = skeletonize(binary.astype(bool))
        return skeleton.astype(np.uint8)