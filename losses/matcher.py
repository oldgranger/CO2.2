import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_convert
from utils.metrics import generalized_box_iou

class HungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 1, cost_bbox: float = 5, cost_giou: float = 2):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.eps = 1e-8

    @torch.no_grad()
    def forward(self, outputs: dict, targets: list) -> list:
        for t in targets:
            boxes = t['boxes']
            assert (boxes[:, 2:] >= boxes[:, :2]).all(), f"Invalid boxes: {boxes}"

        bs, num_queries = outputs['pred_logits'].shape[:2]
        out_prob = outputs['pred_logits'].flatten(0, 1).softmax(-1)
        out_bbox = outputs['pred_boxes'].flatten(0, 1).clamp(min=0, max=1)

        tgt_ids = torch.cat([v['labels'] for v in targets])
        tgt_bbox = torch.cat([v['boxes'] for v in targets]).clamp(min=0, max=1)

        cost_class = -out_prob[:, tgt_ids]
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_giou = -generalized_box_iou(
            box_convert(out_bbox, in_fmt='cxcywh', out_fmt='xyxy'),
            box_convert(tgt_bbox, in_fmt='cxcywh', out_fmt='xyxy')
        ).clamp(min=-1.0, max=1.0)

        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1)
        C = torch.nan_to_num(C, nan=1e8, posinf=1e8, neginf=1e8)

        sizes = [len(v['boxes']) for v in targets]
        indices = [linear_sum_assignment(c[i].cpu()) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
