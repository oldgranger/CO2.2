import matplotlib.pyplot as plt
import torch
from torchvision.transforms import v2 as transforms
from data.dataset import load_datasets
from config import config


def plot_img_with_boxes(img, target):
    """Visualize image with bounding boxes"""
    plt.figure(figsize=(10, 6))

    # Convert tensor to PIL image
    if isinstance(img, torch.Tensor):
        if img.max() <= 1.0:  # Normalized images
            img = transforms.ToPILImage()(img.cpu())
        else:  # Unnormalized
            img = transforms.ToPILImage()(img.byte().cpu())

    plt.imshow(img)

    for box, label in zip(target['boxes'], target['labels']):
        x1, y1, x2, y2 = box.cpu().numpy()
        plt.gca().add_patch(plt.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            fill=False, edgecolor='red', linewidth=2
        ))
        plt.text(x1, y1, config.CLASS_NAMES[label],
                 color='white', backgroundcolor='red')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    # Load datasets
    train_dataset, val_dataset, _ = load_datasets(config.DATA_PATH)

    for i in range(3):
        img, target = val_dataset[i]
        print(f"\nSample {i}:")
        print("Image shape:", img.shape if isinstance(img, torch.Tensor) else img.size)
        print("Boxes:", target['boxes'])
        print("Labels:", target['labels'])
        plot_img_with_boxes(img, target)

    print("First train sample:", train_dataset[0][1])
    print("First val sample:", val_dataset[0][1])