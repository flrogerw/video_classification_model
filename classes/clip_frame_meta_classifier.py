"""
Module: clip_with_metadata_classifier
-------------------------------------
This module defines a PyTorch neural network that combines CLIP image embeddings
with auxiliary metadata for classification. The architecture processes the
embedding and metadata through separate branches before concatenating them
for the final classification.
"""

import torch
from torch import nn
from torch import Tensor
from typing import Optional


class CLIPWithMetadataClassifier(nn.Module):
    """
    A neural network classifier that combines CLIP embeddings with metadata
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
        meta_weight: float = 1.0
    ) -> None:
        """
        Initialize the classifier.

        Args:
            embedding_dim: Dimension of the CLIP embeddings.
            metadata_dim: Dimension of the metadata vector.
            hidden_dim: Number of hidden units in intermediate layers.
            num_classes: Number of output classes.
            meta_weight: Weight to scale the metadata branch output.
        """
        super(CLIPWithMetadataClassifier, self).__init__()
        self.meta_weight = meta_weight

        # Branch to process CLIP embeddings
        self.embedding_branch = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU()
        )

        # Branch to process metadata
        self.metadata_branch = nn.Sequential(
            nn.Linear(metadata_dim, hidden_dim),
            nn.ReLU()
        )

        # Final classification head after concatenating both branches
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
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
            # Ensure float type for computations
            embedding = embedding.float()
            metadata = metadata.float()

            # Process embedding branch
            emb_out = self.embedding_branch(embedding)

            # Process metadata branch and scale by meta_weight
            meta_out = self.metadata_branch(metadata) * self.meta_weight

            # Concatenate processed features
            combined = torch.cat((emb_out, meta_out), dim=1)

            # Final classification output
            return self.classifier(combined)

        except Exception as e:
            # Log the error and return None if something goes wrong
            print(f"[ERROR] Forward pass failed: {e}")
            return None
