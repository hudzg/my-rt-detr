'''
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/util/box_ops.py
'''

import torch
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


import math

def trifocal_ciou_loss(boxes1, boxes2, iout=0.6, eps=1e-7):
    """
    Computes Pairwise Tri-focal CIOU loss between 2 sets of boxes.
    Input: boxes1 [N, 4], boxes2 [M, 4] in (x1, y1, x2, y2) format
    Returns: loss [N, M]
    """
    # N = boxes1.shape[0]
    # M = boxes2.shape[0]
    
    # 1. Basic parameters - Broadcast to [N, M]
    # boxes1 -> [N, 1, 4]
    b1_x1, b1_y1, b1_x2, b1_y2 = boxes1[:, 0].unsqueeze(1), boxes1[:, 1].unsqueeze(1), boxes1[:, 2].unsqueeze(1), boxes1[:, 3].unsqueeze(1)
    # boxes2 -> [1, M, 4]
    b2_x1, b2_y1, b2_x2, b2_y2 = boxes2[:, 0].unsqueeze(0), boxes2[:, 1].unsqueeze(0), boxes2[:, 2].unsqueeze(0), boxes2[:, 3].unsqueeze(0)

    # Intersection area
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * \
                 torch.clamp(inter_rect_y2 - inter_rect_y1, min=0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union_area = (w1 * h1) + (w2 * h2) - inter_area + eps
    
    iou = inter_area / union_area

    # 2. Compute CIOU components
    # Smallest Enclosing Box (C)
    c_x1 = torch.min(b1_x1, b2_x1)
    c_y1 = torch.min(b1_y1, b2_y1)
    c_x2 = torch.max(b1_x2, b2_x2)
    c_y2 = torch.max(b1_y2, b2_y2)
    c2 = (c_x2 - c_x1)**2 + (c_y2 - c_y1)**2 + eps

    # Center distance (rho)
    rho2 = ((b1_x1 + b1_x2 - b2_x1 - b2_x2) ** 2 + \
            (b1_y1 + b1_y2 - b2_y1 - b2_y2) ** 2) / 4

    # Aspect Ratio (v & alpha)
    v = (4 / (math.pi ** 2)) * torch.pow(torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2)
    with torch.no_grad():
        alpha = v / ((1 - iou) + v + eps)

    # CIOU
    ciou = iou - (rho2 / c2) - (alpha * v)
    l_ciou = 1.0 - ciou

    # 3. Apply Tri-focal weights (Equation 5)
    # Default is hard sample (weight = e)
    weights = torch.full_like(iou, math.e)

    # Medium sample: 1 - IOUT < IOU < IOUT
    mask_medium = (iou > (1 - iout)) & (iou < iout)
    weights[mask_medium] = math.exp(1 - iout)

    # Easy sample: IOU >= IOUT
    mask_easy = iou >= iout
    weights[mask_easy] = torch.exp(1 - iou[mask_easy])

    # Final Weighted Loss
    return weights * l_ciou


def diou_loss(boxes1, boxes2, eps=1e-7):
    """
    Computes standard Pairwise DIOU loss between 2 sets of boxes.
    Input: boxes1 [N, 4], boxes2 [M, 4] in (x1, y1, x2, y2) format
    Returns: loss [N, M]
    """
    # 1. Basic parameters - Broadcast to [N, M]
    # boxes1 -> [N, 1, 4]
    b1_x1, b1_y1, b1_x2, b1_y2 = boxes1[:, 0].unsqueeze(1), boxes1[:, 1].unsqueeze(1), boxes1[:, 2].unsqueeze(1), boxes1[:, 3].unsqueeze(1)
    # boxes2 -> [1, M, 4]
    b2_x1, b2_y1, b2_x2, b2_y2 = boxes2[:, 0].unsqueeze(0), boxes2[:, 1].unsqueeze(0), boxes2[:, 2].unsqueeze(0), boxes2[:, 3].unsqueeze(0)

    # Intersection area
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * \
                 torch.clamp(inter_rect_y2 - inter_rect_y1, min=0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union_area = (w1 * h1) + (w2 * h2) - inter_area + eps
    
    iou = inter_area / union_area

    # 2. Compute DIOU components
    # Smallest Enclosing Box (C)
    c_x1 = torch.min(b1_x1, b2_x1)
    c_y1 = torch.min(b1_y1, b2_y1)
    c_x2 = torch.max(b1_x2, b2_x2)
    c_y2 = torch.max(b1_y2, b2_y2)
    c2 = (c_x2 - c_x1)**2 + (c_y2 - c_y1)**2 + eps

    # Center distance (rho)
    rho2 = ((b1_x1 + b1_x2 - b2_x1 - b2_x2) ** 2 + \
            (b1_y1 + b1_y2 - b2_y1 - b2_y2) ** 2) / 4

    # DIOU
    diou = iou - (rho2 / c2)
    
    # Standard DIOU Loss
    l_diou = 1.0 - diou

    return l_diou

def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)