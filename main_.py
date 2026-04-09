import sys
sys.path.insert(0, "/content/drive/MyDrive/fingerprint_project")

from fingerprint.fingerprint_module import FingerprintModule

fm = FingerprintModule()

# ── Full absolute paths — works correctly in Colab ────────────────────────────
BASE         = "/content/drive/MyDrive/fingerprint_project"
IMAGE_DIR    = BASE + "/dataset/raw"
TEMPLATE_DIR = BASE + "/dataset/templates"

# ── Step 1: Enroll all images ─────────────────────────────────────────────────
summary = fm.enroll_batch(
    image_dir    = IMAGE_DIR,
    template_dir = TEMPLATE_DIR
)

print(f"Enrolled : {len(summary['enrolled'])} images")
print(f"Failed   : {len(summary['failed'])} images")

if summary["failed"]:
    for name, reason in summary["failed"]:
        print(f"  FAILED: {name}  ->  {reason}")

# ── Step 2: Verify a probe against a stored template ─────────────────────────
# Change these two filenames to match your actual image files
PROBE_IMAGE    = "user_001_2.bmp"
TEMPLATE_IMAGE = "user_001_1"

result = fm.verify(
    probe_path    = IMAGE_DIR    + "/" + PROBE_IMAGE,
    template_path = TEMPLATE_DIR + "/" + TEMPLATE_IMAGE + ".pkl"
)

print(f"\nScore   : {result.score}")
print(f"Match   : {result.match}")
print(f"Probe minutiae    : {result.metadata.get('probe_minutiae', 'N/A')}")
print(f"Template minutiae : {result.metadata.get('ref_minutiae', 'N/A')}")