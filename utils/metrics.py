import torch

def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    boxes1 = boxes1.float().clamp(min=0.0, max=1.0)
    boxes2 = boxes2.float().clamp(min=0.0, max=1.0)

    boxes1 = torch.stack([
        torch.min(boxes1[:, [0, 2]], dim=1).values,
        torch.min(boxes1[:, [1, 3]], dim=1).values,
        torch.max(boxes1[:, [0, 2]], dim=1).values,
        torch.max(boxes1[:, [1, 3]], dim=1).values
    ], dim=1)

    boxes2 = torch.stack([
        torch.min(boxes2[:, [0, 2]], dim=1).values,
        torch.min(boxes2[:, [1, 3]], dim=1).values,
        torch.max(boxes2[:, [0, 2]], dim=1).values,
        torch.max(boxes2[:, [1, 3]], dim=1).values
    ], dim=1)

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