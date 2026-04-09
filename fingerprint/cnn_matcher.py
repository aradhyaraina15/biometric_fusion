# fingerprint/cnn_matcher.py
import os
import pickle
import numpy as np
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
import cv2

from .cnn_model  import SiameseNetwork
from .config     import (CNN_IMAGE_SIZE, CNN_EMBEDDING_SIZE,
                          CNN_MODEL_PATH, CNN_MATCH_THRESHOLD,
                          CNN_EMBEDDINGS_DIR)
from .models     import FingerprintResult, EmbeddingResult
from .utils      import setup_logger

logger = setup_logger(__name__)


class CNNMatcher:
    """
    Replaces FeatureExtractor + FingerprintMatcher in the CNN pipeline.

    Instead of extracting minutiae and comparing with geometry,
    it encodes each image to a 128-dim vector and compares
    vectors using Euclidean distance.
    """

    def __init__(self, model_path: str = CNN_MODEL_PATH):
        self.device     = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.model      = SiameseNetwork(CNN_EMBEDDING_SIZE).to(self.device)
        self.model_path = model_path
        self._model_loaded = False

    def _ensure_model_loaded(self):
        if not self._model_loaded:
            if not Path(self.model_path).exists():
                raise FileNotFoundError(
                    f"No trained model at {self.model_path}. "
                    f"Run CNNTrainer().train() first."
                )
            self.model.load_state_dict(
                torch.load(self.model_path, map_location=self.device)
            )
            self.model.eval()
            self._model_loaded = True
            logger.info(f"CNN model loaded from {self.model_path}")

    def _preprocess_image(self, img: np.ndarray) -> torch.Tensor:
        """Convert numpy image to normalised CNN input tensor."""
        img = cv2.resize(img, CNN_IMAGE_SIZE)
        img = img.astype(np.float32) / 255.0
        tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        return tensor.to(self.device)

    def encode(self, img: np.ndarray) -> np.ndarray:
        """
        Run image through CNN encoder.
        Returns a 128-dimensional numpy vector.
        """
        self._ensure_model_loaded()
        tensor = self._preprocess_image(img)
        with torch.no_grad():
            embedding = self.model.encode(tensor)
        return embedding.cpu().numpy().flatten()

    def compare(self,
                embedding1: np.ndarray,
                embedding2: np.ndarray) -> float:
        """
        Compare two embedding vectors.
        Returns distance — lower = more similar.
        Converts to a 0–1 similarity score for the fusion layer.
        """
        dist  = float(np.linalg.norm(embedding1 - embedding2))
        # Convert distance to similarity score
        # distance=0 → score=1.0 (identical)
        # distance=large → score→0.0
        score = float(np.exp(-dist))
        return round(score, 4)

    def save_embedding(self,
                        embedding: np.ndarray,
                        user_id: str,
                        embeddings_dir: str = CNN_EMBEDDINGS_DIR):
        """Save a user's embedding vector to disk."""
        Path(embeddings_dir).mkdir(parents=True, exist_ok=True)
        save_path = Path(embeddings_dir) / f"{user_id}.npy"
        np.save(str(save_path), embedding)
        logger.debug(f"Saved embedding for {user_id} → {save_path}")

    def load_embedding(self,
                        user_id: str,
                        embeddings_dir: str = CNN_EMBEDDINGS_DIR) -> np.ndarray:
        """Load a stored embedding vector."""
        load_path = Path(embeddings_dir) / f"{user_id}.npy"
        if not load_path.exists():
            raise FileNotFoundError(f"No embedding for {user_id}")
        return np.load(str(load_path))

    def verify(self,
               probe_img: np.ndarray,
               reference_user_id: str,
               embeddings_dir: str = CNN_EMBEDDINGS_DIR) -> FingerprintResult:
        """
        1-to-1 verification using CNN embeddings.
        """
        probe_emb = self.encode(probe_img)
        ref_emb   = self.load_embedding(reference_user_id, embeddings_dir)
        score     = self.compare(probe_emb, ref_emb)
        match     = score >= CNN_MATCH_THRESHOLD

        return FingerprintResult(
            score          = score,
            match          = match,
            minutiae_count = 0,     # not used in CNN mode
            metadata       = {
                "mode"    : "CNN",
                "user_id" : reference_user_id,
                "distance": round(float(np.linalg.norm(probe_emb - ref_emb)), 4)
            }
        )

    def identify(self,
                  probe_img: np.ndarray,
                  embeddings_dir: str = CNN_EMBEDDINGS_DIR) -> dict:
        """
        1-to-N identification — search entire database.
        """
        embeddings_dir = Path(embeddings_dir)
        stored         = sorted(embeddings_dir.glob("*.npy"))

        if not stored:
            return {"matched": False, "reason": "No embeddings in database"}

        probe_emb = self.encode(probe_img)
        scores    = {}

        for emb_path in stored:
            ref_emb      = np.load(str(emb_path))
            score        = self.compare(probe_emb, ref_emb)
            scores[emb_path.stem] = score

        best_id    = max(scores, key=scores.get)
        best_score = scores[best_id]
        matched    = best_score >= CNN_MATCH_THRESHOLD

        return {
            "matched"   : matched,
            "identity"  : best_id if matched else None,
            "best_score": best_score,
            "all_scores": dict(sorted(scores.items(),
                                       key=lambda x: x[1], reverse=True)),
            "mode"      : "CNN"
        }