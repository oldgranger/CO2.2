from torchvision.transforms import v2 as transforms
from torchvision.tv_tensors import Image
import torch

from config import Config


def get_transform(train: bool):
    transform_list = [
        transforms.ToImage(),  # Converts to tensor and scales to [0,1]
        transforms.ToDtype(torch.float32, scale=True),  # Ensures float32
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),

        # Correct normalization syntax for TorchVision v2:
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ]

    if train:
        transform_list = [
                             transforms.RandomHorizontalFlip(p=0.3),
                             transforms.ColorJitter(
                                 brightness=0.1,
                                 contrast=0.1,
                                 saturation=0.1,
                                 hue=0.02
                             ),
                             transforms.RandomAffine(
                                 degrees=5,
                                 translate=(0.05, 0.05))
                         ] + transform_list

    return transforms.Compose(transform_list)