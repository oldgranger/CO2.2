import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.metrics import generalized_box_iou
from torchvision.ops import box_convert

class SetCriterion(nn.Module):
    def __init__(self, num_classes: int, matcher: nn.Module, weight_dict: dict, eos_coef: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs: dict, targets: list, indices: list, num_boxes: torch.Tensor) -> dict:
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

    def loss_boxes(self, outputs: dict, targets: list, indices: list, num_boxes: torch.Tensor) -> dict:
        src_boxes = outputs['pred_boxes'].clamp(min=0.0, max=1.0)
        idx = self._get_src_permutation_idx(indices)
        src_boxes = src_boxes[idx]

        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_boxes = target_boxes.clamp(min=0.0, max=1.0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none').sum() / num_boxes
        loss_giou = (1 - torch.diag(generalized_box_iou(
            box_convert(src_boxes, in_fmt='cxcywh', out_fmt='xyxy'),
            box_convert(target_boxes, in_fmt='cxcywh', out_fmt='xyxy')
        ))).sum() / num_boxes

        return {'loss_bbox': loss_bbox, 'loss_giou': loss_giou}

    def _get_src_permutation_idx(self, indices: list) -> tuple:
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def forward(self, outputs: dict, targets: list) -> dict:
        if len(targets) == 0:
            device = outputs['pred_logits'].device
            return {
                'loss_ce': torch.tensor(0.0, device=device),
                'loss_bbox': torch.tensor(0.0, device=device),
                'loss_giou': torch.tensor(0.0, device=device)
            }

        indices = self.matcher(outputs, targets)
        num_boxes = sum(len(t['labels']) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float,
                                   device=outputs['pred_logits'].device).clamp(min=1.0)

        losses = {}
        losses.update(self.loss_labels(outputs, targets, indices, num_boxes))
        losses.update(self.loss_boxes(outputs, targets, indices, num_boxes))

        for k in losses.keys():
            if k in self.weight_dict:
                losses[k] *= self.weight_dict[k]

        return losses