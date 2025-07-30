import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
from collections import defaultdict
from config import config


def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter + 1e-6)


@torch.no_grad()
def evaluate_detection_metrics(model, data_loader, device, iou_threshold=0.5):
    """Calculate precision, recall, and classification accuracy"""
    model.eval()
    stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'correct': 0, 'total': 0})

    for images, targets in tqdm(data_loader, desc="Calculating metrics"):
        images = images.to(device)
        outputs = model(images)

        for batch_idx in range(len(images)):
            pred_boxes = outputs['pred_boxes'][batch_idx].cpu().numpy()
            pred_scores = torch.softmax(outputs['pred_logits'][batch_idx], -1).max(-1).values.cpu().numpy()
            pred_labels = outputs['pred_logits'][batch_idx].argmax(-1).cpu().numpy()

            gt_boxes = targets[batch_idx]['boxes'].cpu().numpy()
            gt_labels = targets[batch_idx]['labels'].cpu().numpy()

            # Hungarian matching (simplified version)
            matched = set()
            for i, (pred_box, pred_label, pred_score) in enumerate(zip(pred_boxes, pred_labels, pred_scores)):
                if pred_score < 0.5:  # Confidence threshold
                    continue

                best_iou = -1
                best_gt = -1
                for j, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                    if j in matched:
                        continue
                    iou = calculate_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = j

                if best_iou >= iou_threshold and pred_label == gt_labels[best_gt]:
                    stats[gt_labels[best_gt]]['tp'] += 1
                    stats[gt_labels[best_gt]]['correct'] += 1
                    matched.add(best_gt)
                else:
                    stats[pred_label]['fp'] += 1

                stats[pred_label]['total'] += 1

            # Count unmatched GT as FN
            for j in range(len(gt_boxes)):
                if j not in matched:
                    stats[gt_labels[j]]['fn'] += 1

    # Calculate final metrics
    metrics = {}
    for class_id in range(config.NUM_CLASSES):
        class_name = config.CLASS_NAMES[class_id]
        tp = stats[class_id]['tp']
        fp = stats[class_id]['fp']
        fn = stats[class_id]['fn']

        metrics[class_name] = {
            'precision': tp / (tp + fp + 1e-6),
            'recall': tp / (tp + fn + 1e-6),
            'accuracy': stats[class_id]['correct'] / (stats[class_id]['total'] + 1e-6),
            'f1': 2 * (tp / (tp + fp + 1e-6)) * (tp / (tp + fn + 1e-6)) / (
                        tp / (tp + fp + 1e-6) + tp / (tp + fn + 1e-6) + 1e-6)
        }

    return metrics


def evaluate_coco(model, data_loader, device, coco_json_path):
    """Official COCO evaluation using pycocotools"""
    model.eval()
    results = []
    coco_gt = COCO(coco_json_path)

    for images, _ in tqdm(data_loader, desc="COCO Evaluation"):
        images = images.to(device)
        outputs = model(images)

        for idx, (logits, boxes) in enumerate(zip(outputs['pred_logits'], outputs['pred_boxes'])):
            scores = torch.softmax(logits, -1)[:, :-1].max(-1).values
            keep = scores > 0.5
            boxes = boxes[keep].cpu().numpy()
            scores = scores[keep].cpu().numpy()
            labels = logits[keep].argmax(-1).cpu().numpy()

            for box, score, label in zip(boxes, scores, labels):
                results.append({
                    "image_id": data_loader.dataset.ids[idx],
                    "category_id": int(label) + 1,  # COCO uses 1-based indexing
                    "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                    "score": float(score)
                })

    coco_pred = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return {
        'mAP_50': coco_eval.stats[0],
        'mAP_75': coco_eval.stats[1],
        'mAP': coco_eval.stats[2]
    }


@torch.no_grad()
def evaluate(model, criterion, data_loader, device):
    model.eval()
    criterion.eval()
    total_loss = 0.0

    for images, targets in tqdm(data_loader, desc="Validating"):
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(images)
        loss_dict = criterion(outputs, targets)
        losses = sum(loss_dict[k] * criterion.weight_dict[k] for k in loss_dict.keys())
        total_loss += losses.item()

    return total_loss / len(data_loader)