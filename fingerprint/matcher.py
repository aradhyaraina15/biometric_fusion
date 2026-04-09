# ─────────────────────────────────────────────
#  matcher.py  —  Stage 4: minutiae matching
# ─────────────────────────────────────────────
import numpy as np
from typing import List

from .config import MINUTIAE_DISTANCE_TOLERANCE, MINUTIAE_ANGLE_TOLERANCE
from .models import MinutiaePoint
from .utils  import setup_logger, angle_difference

logger = setup_logger(__name__)


class FingerprintMatcher:

    def match(self,
              probe: List[MinutiaePoint],
              reference: List[MinutiaePoint]) -> float:
        """
        Compare two minutiae sets.
        Returns a similarity score in [0.0, 1.0].
        """
        if not probe or not reference:
            logger.warning("Empty minutiae list — returning score 0.0")
            return 0.0

        matched = self._count_matches(probe, reference)
        score   = matched / max(len(probe), len(reference))
        logger.debug(
            f"Matched {matched} / {max(len(probe), len(reference))} "
            f"→ score {score:.4f}"
        )
        return round(float(score), 4)

    # ── private ───────────────────────────────────────────────────────────────

    def _count_matches(self,
                        probe: List[MinutiaePoint],
                        reference: List[MinutiaePoint]) -> int:
        matched = 0
        used    = set()                          # reference indices already claimed

        for p in probe:
            best_idx  = None
            best_dist = float('inf')

            for i, r in enumerate(reference):
                if i in used:
                    continue
                dist  = float(np.hypot(p.x - r.x, p.y - r.y))
                adiff = angle_difference(p.angle, r.angle)

                if (dist  < MINUTIAE_DISTANCE_TOLERANCE
                        and adiff < MINUTIAE_ANGLE_TOLERANCE
                        and dist  < best_dist):
                    best_dist = dist
                    best_idx  = i

            if best_idx is not None:
                matched += 1
                used.add(best_idx)

        return matched