import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from config import config


def plot_img_with_boxes(img, target, pred=None, save_path=None):
    """Visualize both ground truth and predictions"""
    plt.figure(figsize=(12, 8))
    plt.imshow(T.ToPILImage()(img))
    ax = plt.gca()

    # Plot ground truth
    if 'boxes' in target:
        for box, label in zip(target['boxes'], target['labels']):
            x1, y1, x2, y2 = box.cpu().numpy()
            rect = plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                fill=False, edgecolor='green', linewidth=2, linestyle='--'
            )
            ax.add_patch(rect)
            ax.text(x1, y1 - 5, f"GT: {config.CLASS_NAMES[label]}",
                    color='white', backgroundcolor='green', fontsize=8)

    # Plot predictions
    if pred and 'boxes' in pred:
        for box, label, score in zip(pred['boxes'], pred['labels'], pred['scores']):
            x1, y1, x2, y2 = box.cpu().numpy()
            rect = plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                fill=False, edgecolor='red', linewidth=2
            )
            ax.add_patch(rect)
            ax.text(x1, y1 + 5, f"{config.CLASS_NAMES[label]}: {score:.2f}",
                    color='white', backgroundcolor='red', fontsize=8)

    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


def plot_predictions(model, dataset, device, confidence_threshold=0.5, n=5, save_dir=None):
    """Visualize model predictions with confidence scores"""
    model.eval()
    for i in range(min(n, len(dataset))):
        img, target = dataset[i]
        with torch.no_grad():
            output = model(img.unsqueeze(0).to(device))

        # Process predictions
        scores = torch.softmax(output['pred_logits'][0], -1).max(-1).values
        keep = scores > confidence_threshold
        pred = {
            'boxes': output['pred_boxes'][0][keep].cpu(),
            'labels': output['pred_logits'][0][keep].argmax(-1).cpu(),
            'scores': scores[keep].cpu()
        }

        save_path = f"{save_dir}/pred_{i}.png" if save_dir else None
        plot_img_with_boxes(img, target, pred, save_path)