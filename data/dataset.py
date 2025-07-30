import os
from typing import Optional, List, Dict, Any
import torch
from torchvision.datasets import CocoDetection
from torchvision.transforms import v2 as transforms
from .transforms import get_transform


class CocoDetectionWithTransform(CocoDetection):
    def __init__(self, root: str, annFile: str, transform: Optional[transforms.Compose] = None):
        super().__init__(root, annFile)
        self.transform = transform
        self.base_transform = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True)
        ])
        self._validate_categories()

    def _validate_categories(self):
        valid_ids = {1: "Paper", 2: "Rock", 3: "Scissors"}
        parent_ids = {0}
        for cat in self.coco.cats.values():
            if cat['id'] not in valid_ids and cat['id'] not in parent_ids:
                raise ValueError(
                    f"Unexpected category ID {cat['id']}. Expected: {valid_ids.keys()} or parent IDs {parent_ids}")

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)

        boxes = []
        labels = []
        for obj in target:
            if 'bbox' in obj and 'category_id' in obj:
                category_id = obj['category_id']
                if category_id in {1, 2, 3}:
                    x, y, w, h = obj['bbox']
                    if w > 1 and h > 1:
                        boxes.append([x, y, x + w, y + h])
                        labels.append(category_id - 1)

        # Convert to tensors
        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx])
        }

        if self.transform:
            img, target = self.transform(img, target)
        return img, target


def load_datasets(data_path: str) -> tuple:
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