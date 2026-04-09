# ─────────────────────────────────────────────
#  feature_extractor.py  —  Stage 3: minutiae detection
# ─────────────────────────────────────────────
import numpy as np
from typing import List

from .config import MIN_MINUTIAE_COUNT, BORDER_MARGIN
from .models import MinutiaePoint, MinutiaeType
from .utils  import setup_logger, compute_local_quality

logger = setup_logger(__name__)


class FeatureExtractor:

    def extract(self, skeleton: np.ndarray) -> List[MinutiaePoint]:
        """Detect minutiae from a skeletonized binary image."""
        points = self._detect_minutiae(skeleton)
        points = self._filter_border(points, skeleton.shape)
        logger.debug(f"{len(points)} minutiae after border filter")
        if len(points) < MIN_MINUTIAE_COUNT:
            raise ValueError(
                f"Too few minutiae ({len(points)}) — "
                f"image quality likely too low (need ≥ {MIN_MINUTIAE_COUNT})"
            )
        return points

    # ── private ───────────────────────────────────────────────────────────────

    def _detect_minutiae(self, img: np.ndarray) -> List[MinutiaePoint]:
        points = []
        rows, cols = img.shape
        for y in range(1, rows - 1):
            for x in range(1, cols - 1):
                if img[y, x] == 0:
                    continue                     # background pixel, skip
                cn = self._crossing_number(img, x, y)
                if cn == 1:
                    mtype = MinutiaeType.ENDING
                elif cn == 3:
                    mtype = MinutiaeType.BIFURCATION
                else:
                    continue
                angle   = self._local_orientation(img, x, y)
                quality = compute_local_quality(img, x, y)
                points.append(
                    MinutiaePoint(x=x, y=y, angle=angle,
                                  type=mtype, quality=quality)
                )
        return points

    def _crossing_number(self, img: np.ndarray, x: int, y: int) -> int:
        """8-connected crossing number — counts ridge transitions around pixel."""
        n = [
            int(img[y - 1, x - 1] > 0),
            int(img[y - 1, x    ] > 0),
            int(img[y - 1, x + 1] > 0),
            int(img[y,     x + 1] > 0),
            int(img[y + 1, x + 1] > 0),
            int(img[y + 1, x    ] > 0),
            int(img[y + 1, x - 1] > 0),
            int(img[y,     x - 1] > 0),
        ]
        return sum(abs(n[i] - n[(i + 1) % 8]) for i in range(8)) // 2

    def _local_orientation(self, img: np.ndarray, x: int, y: int) -> float:
        """Estimate ridge orientation via image gradient in a 5×5 patch."""
        h, w = img.shape
        y1, y2 = max(0, y - 2), min(h, y + 3)
        x1, x2 = max(0, x - 2), min(w, x + 3)
        patch = img[y1:y2, x1:x2].astype(np.float32)
        if patch.size == 0:
            return 0.0
        gy, gx = np.gradient(patch)
        return float(np.arctan2(gy.mean(), gx.mean()))

    def _filter_border(self,
                        points: List[MinutiaePoint],
                        shape: tuple) -> List[MinutiaePoint]:
        m = BORDER_MARGIN
        return [
            p for p in points
            if m < p.x < shape[1] - m
            and m < p.y < shape[0] - m
        ]