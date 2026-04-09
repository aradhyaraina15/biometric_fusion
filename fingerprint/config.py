# ─────────────────────────────────────────────
#  config.py  —  all tunable constants live here
# ─────────────────────────────────────────────

# Image dimensions every fingerprint is resized to before processing.
# Both width and height. Change this if your scanner has very different resolution.
IMAGE_SIZE = (300, 300)          # (width, height) — used by cv2.resize
#grid-processing limits values to higher level in a more upward way ...resize cv2 different resolution scanner change this if resized to before preprocessing
# CLAHE contrast enhancement
CLAHE_CLIP_LIMIT  = 1.5         # higher = more aggressive contrast boost
CLAHE_TILE_SIZE   = (8, 8)       # tile grid size for local equalization

# Gabor filter bank frequencies (cycles per pixel).#local equalisation
# More frequencies = slower but more robust preprocessing.#robust planning // gabor frequencies
GABOR_FREQUENCIES = [0.1, 0.15]

# Minutiae detection
MIN_MINUTIAE_COUNT = 10          # reject image if fewer points found after filtering
BORDER_MARGIN      = 20        # pixels — strip this many pixels from each edge

# Matching tolerances
MINUTIAE_DISTANCE_TOLERANCE = 12   # pixels — how far apart two points can be and still match
MINUTIAE_ANGLE_TOLERANCE    = 0.3  # radians (~17°) — how different ridge angles can be

# Decision threshold — score >= this means MATCH
# Lower = more permissive (more false accepts), Higher = stricter (more false rejects)
MATCH_THRESHOLD = 0.35
MIN_MINUTIAE_QUALITY = 0.3

# Local quality window (used in utils.compute_local_quality)
QUALITY_WINDOW = 16


CNN_EMBEDDING_SIZE  = 128      # size of the feature vector the CNN produces
CNN_IMAGE_SIZE      = (96, 96) # CNN input size — smaller than 300x300 for speed
CNN_LEARNING_RATE   = 0.001
CNN_EPOCHS          = 20       # increase if you have more data
CNN_BATCH_SIZE      = 16
CNN_MARGIN          = 1.0      # contrastive loss margin
CNN_MATCH_THRESHOLD = 0.5      # distance below this = same finger

# Paths
CNN_MODEL_PATH      = 'models/siamese_model.pth'
CNN_EMBEDDINGS_DIR  = 'dataset/cnn_embeddings'