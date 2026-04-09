from google.colab import drive
drive.mount('/content/drive')

import sys
import os

# IMPORTANT: Adjust this path to where your 'fingerprint' directory is located in your Google Drive.
# For example, if it's directly under 'My Drive', it would be '/content/drive/My Drive/fingerprint_project_folder'
FINGERPRINT_PATH = '/content/drive/My Drive/fingerprint_project/fingerprint'

# Add the parent directory of 'fingerprint' to sys.path
# This allows Python to find 'fingerprint' as a package within that parent directory.
parent_dir = os.path.dirname(FINGERPRINT_PATH)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# ─────────────────────────────────────────────
#  fingerprint_module.py  —  public API / fusion interface
# ─────────────────────────────────────────────
import pickle
from pathlib import Path
from typing import List

from fingerprint.loader            import FingerprintLoader
from fingerprint.preprocessor      import FingerprintPreprocessor
from fingerprint.feature_extractor import FeatureExtractor
from fingerprint.matcher           import FingerprintMatcher
from fingerprint.models            import MinutiaePoint, FingerprintResult
from fingerprint.config            import MATCH_THRESHOLD
from fingerprint.utils             import setup_logger

logger = setup_logger(__name__)


class FingerprintModule:
    """
    Public API for the fingerprint pipeline.
    Supports both classical (rule-based) and CNN modes.

    Usage:
        fm = FingerprintModule(use_cnn=False)   # classical
        fm = FingerprintModule(use_cnn=True)    # CNN (train first)
    """

    def __init__(self,use_cnn: bool = False):

        self.use_cnn      = use_cnn
        self.loader       = FingerprintLoader()
        self.preprocessor = FingerprintPreprocessor()

        if use_cnn:
            # CNN path — lazy import so torch is only needed when use_cnn=True
            from .cnn_matcher import CNNMatcher
            self.cnn_matcher = CNNMatcher(CNN_MODEL_PATH)
            logger.info("FingerprintModule initialised in CNN mode")
        else:
            # Classical path
            self.extractor = FeatureExtractor()
            self.matcher   = FingerprintMatcher()
            logger.info("FingerprintModule initialised in classical mode")
     # ── Shared internal: load and preprocess ──────────────────────────────────

    def _load_and_preprocess(self, image_path: str) -> np.ndarray:
        """Runs loader + preprocessor — same for both modes."""
        img      = self.loader.load_from_path(image_path)
        skeleton = self.preprocessor.process(img)
        return skeleton
    # ── Enrollment ────────────────────────────────────────────────────────────

    def enroll(self, image_path: str, template_save_path: str):
        """Enroll a fingerprint. Routes to CNN or classical based on use_cnn."""
        logger.info(f"Enrolling [{('CNN' if self.use_cnn else 'classical')}]: "
                    f"{image_path}")

        img = self.loader.load_from_path(image_path)

        if self.use_cnn:
            return self._enroll_cnn(img, image_path, template_save_path)
        else:
            return self._enroll_classical(img, template_save_path)

    def _enroll_classical(self, img, template_save_path: str):
        skeleton = self.preprocessor.process(img)
        features = self.extractor.extract(skeleton)
        Path(template_save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(template_save_path, 'wb') as f:
            pickle.dump(features, f)
        logger.info(f"Classical: {len(features)} minutiae saved")
        return features

    def _enroll_cnn(self, img, image_path: str, template_save_path: str):
        # For CNN: encode image → save embedding as .npy
        skeleton  = self.preprocessor.process(img)
        embedding = self.cnn_matcher.encode(skeleton)

        # Save embedding — use same stem as template_save_path
        user_id   = Path(template_save_path).stem
        self.cnn_matcher.save_embedding(embedding, user_id, CNN_EMBEDDINGS_DIR)

        # Also save path reference as .pkl for compatibility
        Path(template_save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(template_save_path, 'wb') as f:
            pickle.dump({"mode": "CNN", "user_id": user_id}, f)

        logger.info(f"CNN: embedding saved for {user_id}")
        return embedding

    # ── Verification ──────────────────────────────────────────────────────────
 # ── Verification ──────────────────────────────────────────────────────────

    def verify(self, probe_path: str, template_path: str) -> FingerprintResult:
        """1-to-1 match."""
        logger.info(f"Verifying [{('CNN' if self.use_cnn else 'classical')}]: "
                    f"{probe_path}")

        if not Path(template_path).exists():
            raise FileNotFoundError(f"Template not found: {template_path}")

        img = self.loader.load_from_path(probe_path)

        if self.use_cnn:
            skeleton = self.preprocessor.process(img)
            user_id  = Path(template_path).stem
            return self.cnn_matcher.verify(skeleton, user_id, CNN_EMBEDDINGS_DIR)
        else:
            skeleton = self.preprocessor.process(img)
            with open(template_path, 'rb') as f:
                reference_features = pickle.load(f)
            try:
                probe_features = self.extractor.extract(skeleton)
                score          = self.matcher.match(probe_features,
                                                     reference_features)
            except ValueError as e:
                logger.warning(f"Classical verify failed: {e}")
                return FingerprintResult(score=0.0, match=False,
                                          minutiae_count=0,
                                          metadata={"error": str(e)})
            return FingerprintResult(
                score          = score,
                match          = score >= MATCH_THRESHOLD,
                minutiae_count = len(probe_features),
                metadata       = {"mode": "classical",
                                   "ref_minutiae": len(reference_features)}
            )

    # ── Identification ────────────────────────────────────────────────────────

    def identify(self, probe_path: str, template_dir: str) -> dict:
        """1-to-N search across entire database."""
        img      = self.loader.load_from_path(probe_path)
        skeleton = self.preprocessor.process(img)

        if self.use_cnn:
            return self.cnn_matcher.identify(skeleton, CNN_EMBEDDINGS_DIR)
        else:
            # Classical identification
            templates = sorted(Path(template_dir).glob("*.pkl"))
            if not templates:
                return {"matched": False, "reason": "Empty database"}
            try:
                probe_features = self.extractor.extract(skeleton)
            except ValueError as e:
                return {"matched": False, "reason": str(e)}

            scores = {}
            for tpl_path in templates:
                with open(tpl_path, 'rb') as f:
                    ref = pickle.load(f)
                if isinstance(ref, list):   # classical template
                    score = self.matcher.match(probe_features, ref)
                    scores[tpl_path.stem] = score

            if not scores:
                return {"matched": False, "reason": "No classical templates found"}

            best_id    = max(scores, key=scores.get)
            best_score = scores[best_id]
            return {
                "matched"   : best_score >= MATCH_THRESHOLD,
                "identity"  : best_id if best_score >= MATCH_THRESHOLD else None,
                "best_score": best_score,
                "all_scores": dict(sorted(scores.items(),
                                           key=lambda x: x[1], reverse=True)),
                "mode"      : "classical"
            }

    # ── Batch enrollment ──────────────────────────────────────────────────────

    def enroll_batch(self, image_dir: str, template_dir: str) -> dict:
        results   = {"enrolled": [], "failed": []}
        image_dir = Path(image_dir)
        for ext in ("*.bmp", "*.png", "*.tif", "*.jpg"):
            for img_path in sorted(image_dir.glob(ext)):
                tpl_path = Path(template_dir) / (img_path.stem + ".pkl")
                try:
                    self.enroll(str(img_path), str(tpl_path))
                    results["enrolled"].append(img_path.name)
                except Exception as e:
                    logger.warning(f"Failed {img_path.name}: {e}")
                    results["failed"].append((img_path.name, str(e)))
        return results