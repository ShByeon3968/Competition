import torch.nn as nn
from timm import create_model
import torch.nn.functional as F

class ContrastiveClassifier(nn.Module):
    def __init__(self, backbone_name='tf_efficientnet_b4_ns', num_classes=7, feature_dim=128):
        super().__init__()
        self.backbone = create_model(backbone_name, pretrained=True, num_classes=num_classes)
        self.encoder = nn.Sequential(*list(self.backbone.children())[:-1])  # remove classifier
        in_features = self.backbone.classifier.in_features

        self.projector = nn.Sequential(
            nn.Linear(in_features, 512), nn.ReLU(), nn.Linear(512, feature_dim)
        )
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        feat = self.encoder(x).squeeze(-1).squeeze(-1)
        proj = F.normalize(self.projector(feat), dim=1)
        logits = self.classifier(feat)
        return feat, proj, logits