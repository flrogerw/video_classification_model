"""
Module: clip_with_metadata_classifier
-------------------------------------
This module defines a PyTorch neural network that combines CLIP image embeddings
with auxiliary metadata for classification. The architecture processes the
embedding and metadata through deeper branches before concatenating them
for the final classification.
"""

import torch
from torch import nn
from torch import Tensor
from typing import Optional


class CLIPWithMetadataClassifier(nn.Module):
    """
    A deeper neural network classifier that combines CLIP embeddings with metadata
    features for improved classification accuracy.

    Attributes:
        meta_weight (float): Scaling factor applied to the metadata branch output.
        embedding_branch (nn.Sequential): Processes CLIP embeddings.
        metadata_branch (nn.Sequential): Processes metadata features.
        classifier (nn.Sequential): Final classification layers.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        metadata_dim: int = 1,
        hidden_dim: int = 128,
        num_classes: int = 2,
        meta_weight: float = 1.0,
        dropout_prob: float = 0.3
    ) -> None:
        """
        Initialize the classifier.

        Args:
            embedding_dim: Dimension of the CLIP embeddings.
            metadata_dim: Dimension of the metadata vector.
            hidden_dim: Number of hidden units in intermediate layers.
            num_classes: Number of output classes.
            meta_weight: Weight to scale the metadata branch output.
            dropout_prob: Dropout probability for regularization.
        """
        super(CLIPWithMetadataClassifier, self).__init__()
        self.meta_weight = meta_weight

        # Deeper embedding branch with BatchNorm and Dropout
        self.embedding_branch = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
        )

        # Deeper metadata branch with BatchNorm and Dropout
        self.metadata_branch = nn.Sequential(
            nn.Linear(metadata_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
        )

        # Final classifier head - deeper with BatchNorm and Dropout
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + (hidden_dim // 2), hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, embedding: Tensor, metadata: Tensor) -> Optional[Tensor]:
        """
        Forward pass for the model.

        Args:
            embedding: Tensor containing CLIP embeddings (batch_size x embedding_dim).
            metadata: Tensor containing metadata features (batch_size x metadata_dim).

        Returns:
            Output logits tensor (batch_size x num_classes) or None if an error occurs.
        """
        try:
            embedding = embedding.float()
            metadata = metadata.float()

            emb_out = self.embedding_branch(embedding)
            meta_out = self.metadata_branch(metadata) * self.meta_weight

            combined = torch.cat((emb_out, meta_out), dim=1)

            return self.classifier(combined)

        except Exception as e:
            print(f"[ERROR] Forward pass failed: {e}")
            return None
