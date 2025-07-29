from typing import List
import torchvision.transforms as transforms
from torchvision.transforms import v2

def get_transform(train: bool) -> v2.Compose:
    transform_list: List[v2.Transform] = [
        v2.Resize((640, 640)),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

    if train:
        transform_list.extend([
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        ])

    return v2.Compose(transform_list)