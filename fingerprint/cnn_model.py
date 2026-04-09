# fingerprint/cnn_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class FingerprintCNN(nn.Module):
    """
    CNN encoder that takes a fingerprint image and produces
    a 128-dimensional embedding vector.
    Two images with the same finger produce similar vectors.
    Two images with different fingers produce distant vectors.
    """

    def __init__(self, embedding_size: int = 128):
        super(FingerprintCNN, self).__init__()

        # ── Convolutional layers ──────────────────────────────────────────────
        # Each block: Conv → BatchNorm → ReLU → MaxPool
        # Input: (1, 96, 96) — grayscale image

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # (32, 96, 96)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)                               # (32, 48, 48)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # (64, 48, 48)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)                               # (64, 24, 24)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # (128, 24, 24)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)                                # (128, 12, 12)
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1), # (256, 12, 12)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)                                 # (256, 6, 6)
        )

        # ── Fully connected layers ────────────────────────────────────────────
        self.fc = nn.Sequential(
            nn.Flatten(),                          # 256 * 6 * 6 = 9216
            nn.Linear(256 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, embedding_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.fc(x)
        # L2 normalise so all embeddings lie on a unit sphere
        x = F.normalize(x, p=2, dim=1)
        return x


class SiameseNetwork(nn.Module):
    """
    Siamese network — runs two images through the same CNN
    and returns both embeddings plus their distance.
    """

    def __init__(self, embedding_size: int = 128):
        super(SiameseNetwork, self).__init__()
        self.encoder = FingerprintCNN(embedding_size)

    def forward(self,
                img1: torch.Tensor,
                img2: torch.Tensor):
        emb1 = self.encoder(img1)
        emb2 = self.encoder(img2)
        # Euclidean distance between embeddings
        distance = F.pairwise_distance(emb1, emb2)
        return emb1, emb2, distance

    def encode(self, img: torch.Tensor) -> torch.Tensor:
        """Encode a single image to its embedding vector."""
        return self.encoder(img)


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Pulls same-finger pairs together, pushes different-finger pairs apart.

    label = 0 → same finger (genuine pair)   → minimise distance
    label = 1 → different finger (impostor)  → maximise distance up to margin
    """

    def __init__(self, margin: float = 1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self,
                distance: torch.Tensor,
                label: torch.Tensor) -> torch.Tensor:
        genuine_loss  = (1 - label) * distance.pow(2)
        impostor_loss = label * F.relu(self.margin - distance).pow(2)
        loss = (genuine_loss + impostor_loss).mean()
        return loss