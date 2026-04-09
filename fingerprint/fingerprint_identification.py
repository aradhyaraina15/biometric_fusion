# ── Mount Drive ───────────────────────────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

import sys, os, pickle, shutil
from pathlib import Path

# ── Point Python to your project ─────────────────────────────────────────────
PROJECT_ROOT = '/content/drive/MyDrive/fingerprint_project'
sys.path.insert(0, PROJECT_ROOT)

# ── Install libraries ─────────────────────────────────────────────────────────
!pip install opencv-python scikit-image numpy -q

# ── Import YOUR module — this pulls in all 8 files automatically ──────────────
from fingerprint.fingerprint_module import FingerprintModule

# ── Single instance that uses your config.py, models.py, utils.py etc ─────────
fm = FingerprintModule()

# ── Paths — must match what you set up earlier ────────────────────────────────
RAW_DIR      = PROJECT_ROOT + '/dataset/raw'
TEMPLATE_DIR = PROJECT_ROOT + '/dataset/templates'

os.makedirs(RAW_DIR,      exist_ok=True)
os.makedirs(TEMPLATE_DIR, exist_ok=True)

print("All modules loaded. Pipeline ready.")
print(f"Raw images   : {RAW_DIR}")
print(f"Templates    : {TEMPLATE_DIR}")
# ── Cell 2 — Enroll multiple new fingerprints simultaneously ──────────────────

from google.colab import files

# ── Upload multiple images at once ───────────────────────────────────────────
# Hold Ctrl (Windows) or Cmd (Mac) to select multiple files in the picker
print("Upload one or more fingerprint images to add to the database...")
print("Hold Ctrl / Cmd to select multiple files at once.
")
uploaded = files.upload()

if not uploaded:
    print("No files uploaded. Nothing enrolled.")
else:
    # ── Figure out the next available user number in the database ─────────────
    # Looks at existing templates and continues numbering from where it left off
    existing = sorted(Path(TEMPLATE_DIR).glob("*.pkl"))

    # Find the highest existing new_user_XXX number
    existing_nums = []
    for tpl in existing:
        name = tpl.stem                        # e.g. "new_user_007"
        if name.startswith("new_user_"):
            try:
                num = int(name.replace("new_user_", ""))
                existing_nums.append(num)
            except ValueError:
                pass

    next_num = (max(existing_nums) + 1) if existing_nums else 1

    # ── Process each uploaded file separately ─────────────────────────────────
    results = {"enrolled": [], "failed": []}

    for uploaded_filename in sorted(uploaded.keys()):
        file_ext  = os.path.splitext(uploaded_filename)[1].lower()
        user_id   = f"new_user_{next_num:03d}"   # e.g. new_user_001, new_user_002
        img_path  = f"{RAW_DIR}/{user_id}{file_ext}"
        tpl_path  = f"{TEMPLATE_DIR}/{user_id}.pkl"

        # Copy uploaded file from Colab temp space into your raw/ folder
        shutil.copy(f'/content/{uploaded_filename}', img_path)

        # ── Run through your full pipeline ────────────────────────────────────
        # loader.py → preprocessor.py → feature_extractor.py → saves .pkl
        try:
            features = fm.enroll(img_path, tpl_path)
            results["enrolled"].append({
                "original_filename" : uploaded_filename,
                "assigned_id"       : user_id,
                "minutiae"          : len(features),
                "template_path"     : tpl_path,
            })
            next_num += 1    # only increment on success so numbers stay clean

        except Exception as e:
            results["failed"].append({
                "original_filename" : uploaded_filename,
                "reason"            : str(e),
            })

    # ── Summary ───────────────────────────────────────────────────────────────
    print("
" + "=" * 60)
    print(f"  ENROLLMENT COMPLETE")
    print(f"  Enrolled : {len(results['enrolled'])}")
    print(f"  Failed   : {len(results['failed'])}")
    print("=" * 60)

    if results["enrolled"]:
        print(f"
  {'Original filename':<30} {'Assigned ID':<18} {'Minutiae':>8}")
        print(f"  {'-'*30} {'-'*18} {'-'*8}")
        for r in results["enrolled"]:
            print(f"  {r['original_filename']:<30} {r['assigned_id']:<18} {r['minutiae']:>8}")

    if results["failed"]:
        print(f"
Failed enrollments:")
        for r in results["failed"]:
            print(f"  {r['original_filename']:<30}  Reason: {r['reason']}")
        print("Failed images were not added to the database.")
        print("Try a clearer image or lower MIN_MINUTIAE_COUNT in config.py.")
from google.colab import files

# ── Upload the new image ──────────────────────────────────────────────────────
print("Upload the new fingerprint image to add to the database...")
uploaded = files.upload()

if uploaded:
    uploaded_filename = list(uploaded.keys())[0]
    file_ext          = os.path.splitext(uploaded_filename)[1]

    # ── Set the ID for this new fingerprint ───────────────────────────────────
    NEW_USER_ID = 'new_user_001'    # ← CHANGE THIS to your chosen name

    # ── Save image to raw/ ────────────────────────────────────────────────────
    image_path    = f"{RAW_DIR}/{NEW_USER_ID}{file_ext}"
    template_path = f"{TEMPLATE_DIR}/{NEW_USER_ID}.pkl"

    shutil.copy(f'/content/{uploaded_filename}', image_path)

    # ── Call fm.enroll() — this runs your full pipeline ───────────────────────
    # loader.py → preprocessor.py → feature_extractor.py → saves .pkl
    try:
        features = fm.enroll(image_path, template_path)
        print(f"
Enrolled successfully via your pipeline:")
        print(f"  User ID        : {NEW_USER_ID}")
        print(f"  Image path     : {image_path}")
        print(f"  Template saved : {template_path}")
        print(f"  Minutiae stored: {len(features)}")
    except Exception as e:
        print(f"Enrollment failed: {e}")
templates = sorted(Path(TEMPLATE_DIR).glob("*.pkl"))

if not templates:
    print("Database is empty.")
else:
    print(f"{'#':<4} {'User ID':<30} {'Minutiae stored':>16}")
    print(f"{'-'*4} {'-'*30} {'-'*16}")
    for i, tpl in enumerate(templates, 1):
        with open(tpl, 'rb') as f:
            features = pickle.load(f)
        print(f"{i:<4} {tpl.stem:<30} {len(features):>16}")
    print(f"
Total: {len(templates)} fingerprint(s) in database")
    # ── Cell 5 — Remove a fingerprint from the database (interactive) ─────────────

# First show what is currently in the database so user knows what to type
templates = sorted(Path(TEMPLATE_DIR).glob("*.pkl"))

if not templates:
    print("Database is empty. Nothing to remove.")
else:
    print("Current database entries:")
    print(f"  {'#':<4} {'User ID':<30}")
    print(f"  {'-'*4} {'-'*30}")
    for i, tpl in enumerate(templates, 1):
        print(f"  {i:<4} {tpl.stem:<30}")

    print()

    # ── Take input from user ──────────────────────────────────────────────────
    REMOVE_ID = input("Enter the User ID you want to remove: ").strip()

    if not REMOVE_ID:
        print("No input entered. Nothing removed.")
    else:
        tpl_path = Path(TEMPLATE_DIR) / f"{REMOVE_ID}.pkl"

        # ── Remove template ───────────────────────────────────────────────────
        if tpl_path.exists():
            os.remove(tpl_path)
            print(f"
Removed from database : {REMOVE_ID}")
        else:
            print(f"
User ID not found in database: '{REMOVE_ID}'")
            print("Check spelling — it must exactly match one of the names listed above.")

        # ── Also remove raw image if it exists ────────────────────────────────
        removed_image = False
        for ext in ['.png', '.bmp', '.jpg', '.tif']:
            img_path = Path(RAW_DIR) / f"{REMOVE_ID}{ext}"
            if img_path.exists():
                os.remove(img_path)
                print(f"Removed raw image     : {img_path.name}")
                removed_image = True
                break

        if not removed_image and tpl_path.exists() is False:
            print("No raw image found in dataset/raw/ for this user (template only was removed).")

        # ── Show updated database ─────────────────────────────────────────────
        remaining = sorted(Path(TEMPLATE_DIR).glob("*.pkl"))
        print(f"
Database now has {len(remaining)} fingerprint(s) remaining.")