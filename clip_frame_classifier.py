import torch
from torch import nn


class CLIPWithMetadataClassifier(nn.Module):
    def __init__(self, embedding_dim=512, metadata_dim=1, hidden_dim=128, num_classes=2, meta_weight=1.0):
        super(CLIPWithMetadataClassifier, self).__init__()
        self.meta_weight = meta_weight

        self.embedding_branch = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU()
        )

        self.metadata_branch = nn.Sequential(
            nn.Linear(metadata_dim, hidden_dim),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, embedding, metadata):
        embedding = embedding.float()
        metadata = metadata.float()
        emb_out = self.embedding_branch(embedding)
        meta_out = self.metadata_branch(metadata) * self.meta_weight
        combined = torch.cat((emb_out, meta_out), dim=1)
        return self.classifier(combined)
