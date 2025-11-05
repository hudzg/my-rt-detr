import os
import sys
import argparse
from typing import List, Optional

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont

# add project root to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.core import YAMLConfig
from src.data.coco.coco_dataset import (
    mscoco_category2name,
    mscoco_label2category,
)

def build_label_name_lookup():
    # model trả về "label" (0..79). Ta ánh xạ -> category_id -> name
    # name = mscoco_category2name[mscoco_label2category[label]]
    id2name = {}
    for lbl, cat_id in mscoco_label2category.items():
        id2name[lbl] = mscoco_category2name.get(cat_id, str(lbl))
    return id2name

def _measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont):
    try:
        # Pillow >=8.0
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        # Fallback cho Pillow cũ hơn
        if hasattr(font, "getbbox"):
            bbox = font.getbbox(text)
            return bbox[2] - bbox[0], bbox[3] - bbox[1]
        if hasattr(draw, "textsize"):
            return draw.textsize(text, font=font)
        if hasattr(font, "getsize"):
            return font.getsize(text)
        # Fallback cuối
        return max(1, len(text) * 6), 12

def draw_detections(
    image: Image.Image,
    labels: torch.Tensor,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    id2name: dict,
    score_thr: float = 0.5,
    include_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
):
    # to cpu numpy
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu()
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.detach().cpu()
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu()

    # chuẩn hoá dữ liệu
    labels = labels.reshape(-1).to(torch.int64)
    scores = scores.reshape(-1).to(torch.float32)
    boxes = boxes.reshape(-1, 4).to(torch.float32)

    # lọc theo score
    keep = scores >= score_thr

    # map id -> name
    names = [id2name.get(int(l), str(int(l))) for l in labels.tolist()]

    # lọc theo tên nếu có
    if include_names:
        include_set = {n.strip().lower() for n in include_names}
        name_keep = torch.tensor([n.lower() in include_set for n in names], dtype=torch.bool)
        keep = keep & name_keep

    labels = labels[keep]
    scores = scores[keep]
    boxes = boxes[keep]
    names = [n for k, n in zip(keep.tolist(), names) if k]

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for box, name, sc in zip(boxes.tolist(), names, scores.tolist()):
        x0, y0, x1, y1 = box
        draw.rectangle([x0, y0, x1, y1], outline='red', width=2)
        txt = f"{name}: {sc:.2f}"
        # vẽ nền text đơn giản
        tw, th = _measure_text(draw, txt, font)
        draw.rectangle([x0, max(0, y0 - th - 2), x0 + tw + 4, y0], fill='red')
        draw.text((x0 + 2, max(0, y0 - th - 2)), txt, fill='white', font=font)

    if save_path:
        image.save(save_path)
    return image

def main(args):
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cfg = YAMLConfig(args.config, resume=args.resume)
    if not args.resume:
        raise AttributeError('Only support resume to load model.state_dict by now.')

    checkpoint = torch.load(args.resume, map_location='cpu')
    state = checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model']

    # load to train graph then deploy
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model().to(device)
    model.eval()

    # image IO
    im_pil = Image.open(args.im_file).convert('RGB')
    w, h = im_pil.size
    orig_size = torch.tensor([w, h])[None].to(device)

    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
    im_tensor = transforms(im_pil)[None].to(device)

    # autocast an toàn theo device
    use_amp = (device == 'cuda')
    amp_ctx = torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp) if use_amp else torch.no_grad()

    with torch.no_grad():
        with amp_ctx:
            labels, boxes, scores = model(im_tensor, orig_size)

    # lấy phần tử ảnh đầu (batch=1)
    if isinstance(labels, (list, tuple)):
        labels, boxes, scores = labels[0], boxes[0], scores[0]

    # vẽ với tên lớp
    id2name = build_label_name_lookup()
    out_path = args.out if args.out else os.path.splitext(os.path.basename(args.im_file))[0] + '_labeled.jpg'
    include_names = args.classes if args.classes else None

    draw_detections(
        im_pil,
        labels=labels,
        boxes=boxes,
        scores=scores,
        id2name=id2name,
        score_thr=args.score_thr,
        include_names=include_names,
        save_path=out_path
    )

    # In ra console các detection đã giữ lại
    # (name, score, box)
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu()
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.detach().cpu()
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu()

    # lọc giống logic vẽ để in
    keep = scores.reshape(-1) >= args.score_thr
    names_all = [id2name.get(int(l), str(int(l))) for l in labels.reshape(-1).tolist()]
    if include_names:
        include_set = {n.strip().lower() for n in include_names}
        keep = keep & torch.tensor([n.lower() in include_set for n in names_all], dtype=torch.bool)

    labels_kept = labels.reshape(-1)[keep].tolist()
    boxes_kept = boxes.reshape(-1, 4)[keep].tolist()
    scores_kept = scores.reshape(-1)[keep].tolist()
    names_kept = [n for k, n in zip(keep.tolist(), names_all) if k]

    print("Detections:")
    for name, sc, box in zip(names_kept, scores_kept, boxes_kept):
        print(f"- {name}: {sc:.3f} | box={list(map(lambda x: round(float(x), 1), box))}")
    print(f"Saved: {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="RT-DETR inference with class names (COCO)")
    parser.add_argument('-c', '--config', type=str, required=True, help='YAML config path')
    parser.add_argument('-r', '--resume', type=str, required=True, help='Checkpoint path')
    parser.add_argument('-f', '--im-file', type=str, required=True, help='Input image path')
    parser.add_argument('-o', '--out', type=str, default='', help='Output image path')
    parser.add_argument('-d', '--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'], help='Device')
    parser.add_argument('--score-thr', type=float, default=0.5, help='Score threshold for drawing/filtering')
    parser.add_argument('--classes', nargs='*', default=None, help='Only keep these class names (e.g. person car dog). If omitted, keep all.')
    args = parser.parse_args()
    main(args)