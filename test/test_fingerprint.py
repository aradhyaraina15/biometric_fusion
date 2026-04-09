# ─────────────────────────────────────────────
#  tests/test_fingerprint.py
#  Run with:  python -m pytest tests/ -v
#  from the project ROOT (one level above fingerprint/)
# ─────────────────────────────────────────────
import sys
import os
import numpy as np
import pytest

# Make sure the project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fingerprint.models            import MinutiaePoint, MinutiaeType, FingerprintResult
from fingerprint.config            import IMAGE_SIZE, MATCH_THRESHOLD
from fingerprint.utils             import (validate_image_shape,
                                           normalize_image,
                                           angle_difference,
                                           compute_local_quality)
from fingerprint.loader            import FingerprintLoader
from fingerprint.preprocessor      import FingerprintPreprocessor
from fingerprint.feature_extractor import FeatureExtractor
from fingerprint.matcher           import FingerprintMatcher
from fingerprint.fingerprint_module import FingerprintModule


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_synthetic_skeleton(seed: int = 0) -> np.ndarray:
    """
    Generate a deterministic synthetic skeleton image for testing.
    Draws a few horizontal ridge lines so crossing numbers are predictable.
    """
    rng = np.random.default_rng(seed)
    img = np.zeros((300, 300), dtype=np.uint8)
    # Draw 5 horizontal ridge lines
    for row in [60, 90, 130, 170, 210]:
        img[row, 30:270] = 1
    # Add bifurcations by adding short vertical branches
    img[60:90, 100] = 1
    img[130:170, 200] = 1
    return img


def _make_synthetic_gray(seed: int = 0) -> np.ndarray:
    """256-level synthetic grayscale image simulating poor fingerprint contrast."""
    rng = np.random.default_rng(seed)
    img = np.full((300, 300), 128, dtype=np.uint8)
    for row in [60, 90, 130, 170, 210]:
        img[row, 30:270] = 40     # dark ridges
    img += rng.integers(0, 20, img.shape, dtype=np.uint8)
    return img


# ── utils.py tests ────────────────────────────────────────────────────────────

class TestUtils:

    def test_validate_image_shape_valid(self):
        img = np.zeros((300, 300), dtype=np.uint8)
        assert validate_image_shape(img) is True

    def test_validate_image_shape_rejects_none(self):
        assert validate_image_shape(None) is False

    def test_validate_image_shape_rejects_3d(self):
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        assert validate_image_shape(img) is False

    def test_validate_image_shape_rejects_empty(self):
        img = np.zeros((0, 300), dtype=np.uint8)
        assert validate_image_shape(img) is False

    def test_normalize_image_range(self):
        img = np.array([[0, 128, 255]], dtype=np.uint8)
        norm = normalize_image(img)
        assert float(norm.min()) == pytest.approx(0.0, abs=1e-5)
        assert float(norm.max()) == pytest.approx(1.0, abs=1e-5)

    def test_normalize_flat_image(self):
        img = np.full((10, 10), 128, dtype=np.uint8)
        norm = normalize_image(img)
        assert norm.sum() == 0.0    # flat → all zeros

    def test_angle_difference_same(self):
        assert angle_difference(1.0, 1.0) == pytest.approx(0.0, abs=1e-6)

    def test_angle_difference_wraparound(self):
        import math
        # 0.1 and 2π-0.1 should be ~0.2 apart, not 2π-0.2
        diff = angle_difference(0.1, 2 * math.pi - 0.1)
        assert diff == pytest.approx(0.2, abs=1e-5)

    def test_compute_local_quality_range(self):
        img = _make_synthetic_gray()
        q = compute_local_quality(img, 100, 100)
        assert 0.0 <= q <= 1.0


# ── loader.py tests ───────────────────────────────────────────────────────────

class TestLoader:

    def test_load_from_array_grayscale(self):
        loader = FingerprintLoader()
        arr = np.random.randint(0, 256, (200, 200), dtype=np.uint8)
        out = loader.load_from_array(arr)
        assert out.shape == (IMAGE_SIZE[1], IMAGE_SIZE[0])

    def test_load_from_array_color_converts(self):
        loader = FingerprintLoader()
        arr = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        out = loader.load_from_array(arr)
        assert out.ndim == 2

    def test_load_from_array_rejects_none(self):
        loader = FingerprintLoader()
        with pytest.raises(ValueError):
            loader.load_from_array(None)

    def test_load_from_path_missing_file(self):
        loader = FingerprintLoader()
        with pytest.raises(ValueError, match="Could not read"):
            loader.load_from_path("this_file_does_not_exist.png")


# ── preprocessor.py tests ─────────────────────────────────────────────────────

class TestPreprocessor:

    def test_process_returns_binary_uint8(self):
        pp  = FingerprintPreprocessor()
        img = _make_synthetic_gray()
        out = pp.process(img)
        assert out.dtype == np.uint8
        unique = set(np.unique(out))
        assert unique.issubset({0, 1})   # skeleton is 0/1

    def test_process_output_shape_matches_input(self):
        pp  = FingerprintPreprocessor()
        img = _make_synthetic_gray()
        out = pp.process(img)
        assert out.shape == img.shape


# ── feature_extractor.py tests ───────────────────────────────────────────────

class TestFeatureExtractor:

    def test_extract_returns_list(self):
        fe  = FeatureExtractor()
        skl = _make_synthetic_skeleton()
        pts = fe.extract(skl)
        assert isinstance(pts, list)

    def test_all_points_in_bounds(self):
        fe  = FeatureExtractor()
        skl = _make_synthetic_skeleton()
        pts = fe.extract(skl)
        for p in pts:
            assert 0 <= p.x < skl.shape[1]
            assert 0 <= p.y < skl.shape[0]

    def test_quality_in_range(self):
        fe  = FeatureExtractor()
        skl = _make_synthetic_skeleton()
        pts = fe.extract(skl)
        for p in pts:
            assert 0.0 <= p.quality <= 1.0

    def test_type_is_enum(self):
        fe  = FeatureExtractor()
        skl = _make_synthetic_skeleton()
        pts = fe.extract(skl)
        for p in pts:
            assert isinstance(p.type, MinutiaeType)

    def test_low_quality_image_raises(self):
        fe  = FeatureExtractor()
        # Blank skeleton has no minutiae → should raise ValueError
        skl = np.zeros((300, 300), dtype=np.uint8)
        with pytest.raises(ValueError, match="Too few minutiae"):
            fe.extract(skl)


# ── matcher.py tests ──────────────────────────────────────────────────────────

class TestMatcher:

    def _make_point(self, x, y, angle=0.0):
        return MinutiaePoint(x=x, y=y, angle=angle,
                             type=MinutiaeType.ENDING, quality=1.0)

    def test_identical_sets_score_one(self):
        m  = FingerprintMatcher()
        pts = [self._make_point(50, 50), self._make_point(100, 100),
               self._make_point(150, 150), self._make_point(200, 200),
               self._make_point(250, 250)]
        assert m.match(pts, pts) == pytest.approx(1.0, abs=1e-4)

    def test_empty_probe_returns_zero(self):
        m   = FingerprintMatcher()
        ref = [self._make_point(50, 50)]
        assert m.match([], ref) == 0.0

    def test_completely_different_sets_score_low(self):
        m  = FingerprintMatcher()
        a  = [self._make_point(x, 50) for x in range(10, 260, 50)]
        b  = [self._make_point(x, 250) for x in range(10, 260, 50)]
        assert m.match(a, b) < 0.2

    def test_score_in_valid_range(self):
        m  = FingerprintMatcher()
        a  = [self._make_point(50 + i * 10, 100) for i in range(5)]
        b  = [self._make_point(52 + i * 10, 101) for i in range(5)]
        s  = m.match(a, b)
        assert 0.0 <= s <= 1.0


# ── end-to-end pipeline test ──────────────────────────────────────────────────

class TestEndToEnd:
    """
    Runs the full pipeline on synthetic images without needing real scans.
    Replace _make_synthetic_gray() with a real image path for live testing.
    """

    def test_full_pipeline_same_image(self, tmp_path):
        fm       = FingerprintModule()
        img      = _make_synthetic_gray()

        # Save synthetic image as a PNG so the loader can read it
        import cv2
        img_path = str(tmp_path / "test_finger.png")
        cv2.imwrite(img_path, img)

        tpl_path  = str(tmp_path / "test_finger.pkl")
        probe_path = img_path                      # same image = should match

        fm.enroll(img_path, tpl_path)
        result = fm.verify(probe_path, tpl_path)

        assert isinstance(result, FingerprintResult)
        assert 0.0 <= result.score <= 1.0

    def test_verify_missing_template_raises(self, tmp_path):
        fm = FingerprintModule()
        with pytest.raises(FileNotFoundError):
            fm.verify("any.png", str(tmp_path / "nonexistent.pkl"))