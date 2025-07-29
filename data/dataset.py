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

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        img = self.base_transform(img)

        boxes = []
        labels = []
        for obj in target:
            if 'bbox' in obj and 'category_id' in obj:
                category_id = obj['category_id']
                if category_id in [1, 2, 3]:
                    x, y, w, h = obj['bbox']
                    if w > 1 and h > 1:
                        boxes.append([x, y, x + w, y + h])
                        labels.append(category_id - 1)

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