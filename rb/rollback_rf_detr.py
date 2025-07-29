from typing import List

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms
from torchvision.datasets import CocoDetection
import os
import torch.nn as nn
from torchvision.ops import box_iou, box_convert
from scipy.optimize import linear_sum_assignment
import torchvision


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


## 1. Data Preparation
class CocoDetectionWithTransform(CocoDetection):
    def __init__(self, root, annFile, transform=None):
        super().__init__(root, annFile)
        self.transform = transform
        # Add initial conversion transforms
        self.base_transform = transforms.Compose([
            transforms.ToImage(),  # Converts PIL to tensor
            transforms.ToDtype(torch.float32, scale=True)  # Converts to float32
        ])

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)

        # Convert PIL to tensor first
        img = self.base_transform(img)

        # Process targets
        boxes = []
        labels = []
        for obj in target:
            if 'bbox' in obj and 'category_id' in obj:
                x, y, w, h = obj['bbox']
                if w > 1 and h > 1:  # Filter tiny boxes
                    boxes.append([x, y, x + w, y + h])  # xywh to xyxy
                    labels.append(obj['category_id'])

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        target = {'boxes': boxes, 'labels': labels}

        if self.transform:
            img, target = self.transform(img, target)

        return img, target


def get_transform(train: bool) -> transforms.Compose:
    transform_list: List[transforms.Transform] = [
        transforms.Resize((640, 640)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

    if train:
        transform_list.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        ])

    return transforms.Compose(transform_list)


def load_datasets(data_path: str):
    train_dataset = CocoDetectionWithTransform(
        root=os.path.join(data_path, 'train'),
        annFile=os.path.join(data_path, 'annotations', 'instances_train.json'),
        transform=get_transform(train=True)
    )

    val_dataset = CocoDetectionWithTransform(
        root=os.path.join(data_path, 'val'),
        annFile=os.path.join(data_path, 'annotations', 'instances_val.json'),
        transform=get_transform(train=False)
    )

    test_dataset = CocoDetectionWithTransform(
        root=os.path.join(data_path, 'test'),
        annFile=os.path.join(data_path, 'annotations', 'instances_test.json'),
        transform=get_transform(train=False)
    )

    return train_dataset, val_dataset, test_dataset


def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    images = torch.stack(images, dim=0)
    return images, targets


def create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size=2):
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=True
    )

    return train_loader, val_loader, test_loader


## 2. Model Implementation
class ReceptiveFieldEnhancement(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=3, dilation=3)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

        # Initialize weights
        for conv in [self.conv1, self.conv2, self.conv3]:
            nn.init.xavier_uniform_(conv.weight)
            nn.init.constant_(conv.bias, 0)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)

        min_h = min(x1.size(2), x2.size(2), x3.size(2))
        min_w = min(x1.size(3), x2.size(3), x3.size(3))

        x1 = F.interpolate(x1, size=(min_h, min_w), mode='bilinear', align_corners=False)
        x2 = F.interpolate(x2, size=(min_h, min_w), mode='bilinear', align_corners=False)
        x3 = F.interpolate(x3, size=(min_h, min_w), mode='bilinear', align_corners=False)

        out = self.norm(x1 + x2 + x3)
        return self.activation(out)


class RFDETR(nn.Module):
    def __init__(self, num_classes, hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()

        # Store parameters
        self.hidden_dim = hidden_dim
        self.nheads = nheads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_classes = num_classes

        # Backbone initialization
        backbone = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)

        # Freeze early layers
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

        # Initialize bbox head layers
        for layer in self.bbox_head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)

        # Positional embeddings
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

        # Initialize embeddings
        nn.init.uniform_(self.query_pos, -0.1, 0.1)
        nn.init.uniform_(self.row_embed, -0.1, 0.1)
        nn.init.uniform_(self.col_embed, -0.1, 0.1)

    def forward(self, x):
        # Backbone feature extraction
        features = self.backbone(x)  # [batch, 2048, H, W]

        # Receptive field enhancement
        features = self.rfe(features)  # [batch, hidden_dim, H', W']
        h = self.conv(features)  # [batch, hidden_dim, H', W']

        # Positional encoding
        H, W = h.shape[-2:]
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1).to(h.device)

        # Prepare transformer inputs
        h = h.flatten(2).permute(2, 0, 1)  # [H'*W', batch, hidden_dim]

        # Query positions
        query_pos = self.query_pos.unsqueeze(1).repeat(1, x.size(0), 1)  # [100, batch, hidden_dim]

        # Transformer
        h = self.transformer(pos + h, query_pos)  # [100, batch, hidden_dim]
        h = h.transpose(0, 1)  # [batch, 100, hidden_dim]

        # Predictions
        outputs_class = self.linear_class(h)  # [batch, 100, num_classes+1]
        outputs_coord = self.bbox_head(h).sigmoid().clamp(min=0.0, max=1.0)  # [batch, 100, 4]

        return {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}


## 3. Loss and Training
class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=1, cost_bbox=5, cost_giou=2):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.eps = 1e-8

    @torch.no_grad()
    def forward(self, outputs, targets):

        for t in targets:
            boxes = t['boxes']
            assert (boxes[:, 2:] >= boxes[:, :2]).all(), f"Invalid boxes: {boxes}"

        bs, num_queries = outputs['pred_logits'].shape[:2]

        out_prob = outputs['pred_logits'].flatten(0, 1).softmax(-1)
        out_bbox = outputs['pred_boxes'].flatten(0, 1).clamp(min=0, max=1)

        tgt_ids = torch.cat([v['labels'] for v in targets])
        tgt_bbox = torch.cat([v['boxes'] for v in targets]).clamp(min=0, max=1)

        # Compute classification cost (stable)
        cost_class = -out_prob[:, tgt_ids]

        # Compute L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute GIoU cost (stable)
        giou = generalized_box_iou(
            box_convert(out_bbox, in_fmt='cxcywh', out_fmt='xyxy'),
            box_convert(tgt_bbox, in_fmt='cxcywh', out_fmt='xyxy')
        )
        cost_giou = -giou.clamp(min=-1.0, max=1.0)

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1)

        # Handle NaN/Inf
        C = torch.nan_to_num(C, nan=1e8, posinf=1e8, neginf=1e8)

        sizes = [len(v['boxes']) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def generalized_box_iou(boxes1, boxes2):
    # Convert to float32 for stability
    boxes1 = boxes1.float().clamp(min=0.0, max=1.0)
    boxes2 = boxes2.float().clamp(min=0.0, max=1.0)

    # Ensure valid boxes (x2 >= x1, y2 >= y1)
    boxes1 = torch.stack([
        torch.min(boxes1[:, [0, 2]], dim=1).values,  # x1
        torch.min(boxes1[:, [1, 3]], dim=1).values,  # y1
        torch.max(boxes1[:, [0, 2]], dim=1).values,  # x2
        torch.max(boxes1[:, [1, 3]], dim=1).values  # y2
    ], dim=1)

    boxes2 = torch.stack([
        torch.min(boxes2[:, [0, 2]], dim=1).values,
        torch.min(boxes2[:, [1, 3]], dim=1).values,
        torch.max(boxes2[:, [0, 2]], dim=1).values,
        torch.max(boxes2[:, [1, 3]], dim=1).values
    ], dim=1)

    # Calculate areas
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    iou = inter / (union + 1e-8)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    convex_area = wh[:, :, 0] * wh[:, :, 1] + 1e-8

    giou = iou - (convex_area - union) / convex_area
    return giou.clamp(min=-1.0, max=1.0)


class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, eos_coef=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes):
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)])

        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2),
            target_classes,
            weight=self.empty_weight,
            reduction='none'
        )
        return {'loss_ce': loss_ce.sum() / num_boxes}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        src_boxes = outputs['pred_boxes'].clamp(min=0.0, max=1.0)
        idx = self._get_src_permutation_idx(indices)
        src_boxes = src_boxes[idx]

        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_boxes = target_boxes.clamp(min=0.0, max=1.0)

        # L1 loss
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none').sum() / num_boxes

        # GIoU loss
        giou = generalized_box_iou(
            box_convert(src_boxes, in_fmt='cxcywh', out_fmt='xyxy'),
            box_convert(target_boxes, in_fmt='cxcywh', out_fmt='xyxy')
        )
        loss_giou = (1 - torch.diag(giou)).sum() / num_boxes

        return {'loss_bbox': loss_bbox, 'loss_giou': loss_giou}

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def forward(self, outputs, targets):
        # Handle empty targets case
        if len(targets) == 0:
            device = outputs['pred_logits'].device
            return {
                'loss_ce': torch.tensor(0.0, device=device),
                'loss_bbox': torch.tensor(0.0, device=device),
                'loss_giou': torch.tensor(0.0, device=device)
            }

        indices = self.matcher(outputs, targets)
        num_boxes = sum(len(t['labels']) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=outputs['pred_logits'].device).clamp(min=1.0)

        losses = {}
        losses.update(self.loss_labels(outputs, targets, indices, num_boxes))
        losses.update(self.loss_boxes(outputs, targets, indices, num_boxes))

        # Apply loss weights
        for k in losses.keys():
            if k in self.weight_dict:
                losses[k] *= self.weight_dict[k]

        return losses


from tqdm import tqdm
import torch.nn.functional as F

def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, print_freq=10):
    model.train()
    criterion.train()

    running_loss = 0.0
    progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Epoch {epoch}")

    for batch_idx, (samples, targets) in progress_bar:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)

        if any(torch.isnan(v) or torch.isinf(v) for v in loss_dict.values()):
            print(f"⚠️  NaN in batch {batch_idx}:")
            for k, v in loss_dict.items():
                print(f"  {k}: {v}")
            continue

        losses = sum(loss_dict.values())
        loss_value = losses.item()

        if not torch.isfinite(losses):
            print(f"❌ Loss is not finite at batch {batch_idx}. Skipping.")
            continue

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += loss_value
        if batch_idx % print_freq == 0:
            progress_bar.set_postfix(loss=loss_value)

    avg_loss = running_loss / len(data_loader)
    print(f"✅ Epoch {epoch} finished. Average Loss: {avg_loss:.4f}")


@torch.no_grad()
def evaluate(model, criterion, data_loader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)
            loss_dict = criterion(outputs, targets)
            losses = sum(loss_dict[k] * criterion.weight_dict[k] for k in loss_dict.keys())
            total_loss += losses.item()

    return total_loss / len(data_loader)


def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=10, lr=1e-4):
    model.to(device)
    best_loss = float('inf')

    # Weight dict for balanced losses
    weight_dict = {'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2}
    criterion.weight_dict = weight_dict

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass
            outputs = model(images)
            loss_dict = criterion(outputs, targets)

            # Skip batch if loss is invalid
            if any(torch.isnan(v) or torch.isinf(v) for v in loss_dict.values()):
                print(f"Skipping batch {batch_idx} due to invalid loss")
                continue

            # Total loss
            losses = sum(loss_dict.values())

            # Backward pass with gradient clipping
            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += losses.item()

            # Log every 10 batches
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {losses.item():.4f}")

        # Validation
        val_loss = evaluate(model, criterion, val_loader, device)
        scheduler.step()

        print(f"Epoch {epoch} | Train Loss: {epoch_loss / len(train_loader):.4f} | Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

    return model

## Main Execution
def main():
    # Load data
    data_path = '../datasets/data'
    train_dataset, val_dataset, test_dataset = load_datasets(data_path)
    train_loader, val_loader, test_loader = create_data_loaders(train_dataset, val_dataset, test_dataset)

    # Initialize model
    num_classes = 91  # COCO classes
    model = RFDETR(num_classes=num_classes).to(device)
    for param in model.parameters():
        param.requires_grad = True

    # Initialize matcher and criterion
    matcher = HungarianMatcher(cost_class=1, cost_bbox=1, cost_giou=1)
    weight_dict = {'loss_ce': 1, 'loss_bbox': 1, 'loss_giou': 1}
    criterion = SetCriterion(num_classes, matcher, weight_dict, eos_coef=0.1).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # Debug: Test forward pass
    print("Testing forward pass with dummy data...")
    dummy_input = torch.randn(2, 3, 640, 640).to(device)
    dummy_output = model(dummy_input)
    print("Dummy forward pass successful!")

    # Train model
    print("Starting training...")
    model = train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=10)

    print("Training completed!")

    torch.save(model.state_dict(), "rfdetr_final.pth")
    print("✅ Model weights saved to rfdetr_final.pth")

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()