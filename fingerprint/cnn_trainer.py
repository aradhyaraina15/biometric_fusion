import os
import pickle
import itertools
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2

from .cnn_model  import SiameseNetwork, ContrastiveLoss
from .config     import (CNN_IMAGE_SIZE, CNN_EMBEDDING_SIZE, CNN_LEARNING_RATE,
                          CNN_EPOCHS, CNN_BATCH_SIZE, CNN_MARGIN, CNN_MODEL_PATH)
from .utils      import setup_logger

logger = setup_logger(__name__)


class FingerprintPairDataset(Dataset):
    """
    Builds BALANCED pairs of fingerprint images for Siamese training.
    Genuine pair (same user)  → label 0
    Impostor pair (diff user) → label 1
    Balanced so genuine count equals impostor count.
    Images are pre-loaded into RAM at init so Drive is only read once.
    """

    def __init__(self, image_dir: str, img_size: tuple = CNN_IMAGE_SIZE):
        self.img_size   = img_size
        self.pairs      = []
        self.img_cache  = {}    # pre-load all images into RAM
        self._build_pairs(image_dir)

    def _load_image_to_tensor(self, path: str) -> torch.Tensor:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Cannot load: {path}")
        img = cv2.resize(img, self.img_size)
        img = img.astype(np.float32) / 255.0
        return torch.tensor(img).unsqueeze(0)   # (1, H, W)

    def _build_pairs(self, image_dir: str):
        image_dir = Path(image_dir)
        images    = []
        for ext in ("*.png", "*.bmp", "*.jpg", "*.tif"):
            images.extend(image_dir.glob(ext))
        images = sorted(images)

        if not images:
            raise ValueError(f"No images found in {image_dir}")

        # Pre-load ALL images into RAM — avoids re-reading files each epoch
        print(f"Pre-loading {len(images)} images into RAM...")
        for img_path in images:
            self.img_cache[str(img_path)] = self._load_image_to_tensor(
                str(img_path)
            )
        print(f"All images cached in RAM.")

        # Group by user ID — user_101_1.png → user_101
        groups = defaultdict(list)
        for img_path in images:
            parts   = img_path.stem.rsplit("_", 1)
            user_id = parts[0] if len(parts) == 2 else img_path.stem
            groups[user_id].append(str(img_path))

        user_ids = list(groups.keys())

        # Genuine pairs
        genuine_pairs = []
        for user_id, paths in groups.items():
            for p1, p2 in itertools.combinations(paths, 2):
                genuine_pairs.append((p1, p2, 0))

        # Impostor pairs
        impostor_pairs = []
        for u1, u2 in itertools.combinations(user_ids, 2):
            p1 = groups[u1][0]
            p2 = groups[u2][0]
            impostor_pairs.append((p1, p2, 1))

        # Balance — match counts so CNN sees equal genuine and impostor
        min_count = min(len(genuine_pairs), len(impostor_pairs))
        rng       = np.random.default_rng(42)

        genuine_sampled  = rng.choice(
            len(genuine_pairs), size=min_count, replace=False
        ).tolist()
        impostor_sampled = rng.choice(
            len(impostor_pairs), size=min_count, replace=False
        ).tolist()

        self.pairs = (
            [genuine_pairs[i]  for i in genuine_sampled] +
            [impostor_pairs[i] for i in impostor_sampled]
        )

        # Shuffle
        rng.shuffle(self.pairs)

        logger.info(
            f"Balanced dataset: {len(self.pairs)} pairs "
            f"({min_count} genuine + {min_count} impostor)"
        )
        print(f"Balanced dataset: {min_count} genuine + {min_count} impostor "
              f"= {len(self.pairs)} total pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        path1, path2, label = self.pairs[idx]
        img1  = self.img_cache[path1]
        img2  = self.img_cache[path2]
        return img1, img2, torch.tensor(float(label))


class CNNTrainer:

    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"Using device: {self.device}")

        if self.device.type == "cpu":
            print("WARNING: GPU not detected. Training on CPU will be slow.")
            print("Go to Runtime → Change runtime type → T4 GPU → Save")
        else:
            print(f"GPU detected: {torch.cuda.get_device_name(0)}")

        self.model     = SiameseNetwork(CNN_EMBEDDING_SIZE).to(self.device)
        self.criterion = ContrastiveLoss(CNN_MARGIN)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=CNN_LEARNING_RATE,
            weight_decay=1e-4    # slight regularisation to prevent overfitting
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            patience=3,          # reduce LR if no improvement for 3 epochs
            factor=0.5)

    def train(self,
              image_dir:       str,
              model_save_path: str = CNN_MODEL_PATH) -> float:
        """
        Train Siamese CNN on images in image_dir.
        Uses balanced pairs, pre-loaded images, and GPU if available.
        Saves best model weights to model_save_path.
        """
        # ── Build dataset ─────────────────────────────────────────────────────
        dataset    = FingerprintPairDataset(image_dir)
        dataloader = DataLoader(
            dataset,
            batch_size  = CNN_BATCH_SIZE,
            shuffle     = True,
            num_workers = 0,        # 0 required for Colab
            pin_memory  = (self.device.type == "cuda")
        )

        Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
        best_loss  = float("inf")
        no_improve = 0
        patience   = 6             # stop early if no improvement for 6 epochs

        print(f"\nTraining on {len(dataset)} balanced pairs...")
        print(f"Device     : {self.device}")
        print(f"Batch size : {CNN_BATCH_SIZE}")
        print(f"Epochs     : {CNN_EPOCHS}")
        print(f"\n{'Epoch':<8} {'Loss':>10} {'LR':>12} {'Status':>10}")
        print("-" * 44)

        for epoch in range(1, CNN_EPOCHS + 1):
            self.model.train()
            epoch_loss = 0.0
            n_batches  = 0

            for img1, img2, label in dataloader:
                # Move to GPU if available
                img1  = img1.to(self.device,  non_blocking=True)
                img2  = img2.to(self.device,  non_blocking=True)
                label = label.to(self.device, non_blocking=True)

                self.optimizer.zero_grad(set_to_none=True)
                _, _, distance = self.model(img1, img2)
                loss           = self.criterion(distance, label)
                loss.backward()

                # Gradient clipping — prevents exploding gradients
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0
                )

                self.optimizer.step()
                epoch_loss += loss.item()
                n_batches  += 1

            avg_loss   = epoch_loss / n_batches
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step(avg_loss)

            if avg_loss < best_loss:
                best_loss  = avg_loss
                no_improve = 0
                torch.save(self.model.state_dict(), model_save_path)
                status = "saved"
            else:
                no_improve += 1
                status = f"no improve {no_improve}/{patience}"

            print(f"{epoch:<8} {avg_loss:>10.4f} "
                  f"{current_lr:>12.6f} {status:>10}")

            # Early stopping
            if no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"— no improvement for {patience} epochs.")
                break

        print(f"\nTraining complete.")
        print(f"Best loss  : {best_loss:.4f}")
        print(f"Model saved: {model_save_path}")
        return best_loss

    def load_trained_model(self,
                            model_path: str = CNN_MODEL_PATH) -> SiameseNetwork:
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"No trained model at {model_path}. Run train() first."
            )
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.model.eval()
        logger.info(f"Loaded model from {model_path}")
        return self.model