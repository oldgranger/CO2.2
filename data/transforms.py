import torch
from torchvision.transforms import v2 as transforms
from torchvision.transforms.v2 import Transform

def get_transform(train: bool) -> Transform:
    transform_list = [
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]

    if train:
        transform_list += [
            transforms.RandomPhotometricDistort(),
            # transforms.RandomIoUCrop()
            transforms.RandomZoomOut(fill=0),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        ]

    transform_list.append(transforms.Resize((640, 640)))

    return transforms.Compose(transform_list)