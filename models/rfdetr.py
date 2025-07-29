import torch
import torch.nn as nn
import torchvision
from .rfe import ReceptiveFieldEnhancement

class RFDETR(nn.Module):
    def __init__(self, num_classes: int, hidden_dim: int = 256, nheads: int = 8,
                 num_encoder_layers: int = 6, num_decoder_layers: int = 6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.nheads = nheads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_classes = num_classes

        # Backbone
        backbone = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        for name, param in backbone.named_parameters():
            if 'layer1' in name or 'layer2' in name or 'bn' in name:
                param.requires_grad_(False)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])

        # Receptive Field Enhancement
        self.rfe = ReceptiveFieldEnhancement(in_channels=2048, out_channels=hidden_dim)

        # Projection layer
        self.conv = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1)
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 0)

        # Transformer
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=nheads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=False
        )

        # Classification head
        self.linear_class = nn.Linear(in_features=hidden_dim, out_features=num_classes + 1)
        nn.init.xavier_uniform_(self.linear_class.weight)
        nn.init.constant_(self.linear_class.bias, 0)

        # Bounding box regression head
        self.bbox_head = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=4)
        )
        for layer in self.bbox_head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)

        # Positional embeddings
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        nn.init.uniform_(self.query_pos, -0.1, 0.1)
        nn.init.uniform_(self.row_embed, -0.1, 0.1)
        nn.init.uniform_(self.col_embed, -0.1, 0.1)

    def forward(self, x: torch.Tensor) -> dict:
        features = self.backbone(x)
        features = self.rfe(features)
        h = self.conv(features)

        H, W = h.shape[-2:]
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1).to(h.device)

        h = h.flatten(2).permute(2, 0, 1)
        query_pos = self.query_pos.unsqueeze(1).repeat(1, x.size(0), 1)
        h = self.transformer(pos + h, query_pos).transpose(0, 1)

        outputs_class = self.linear_class(h)
        outputs_coord = self.bbox_head(h).sigmoid().clamp(min=0.0, max=1.0)

        return {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}