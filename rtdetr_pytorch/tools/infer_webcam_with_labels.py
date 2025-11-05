import os
import sys
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont

# Thêm project root vào sys.path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.core import YAMLConfig
from src.data.coco.coco_dataset import (
    mscoco_category2name,
    mscoco_label2category,
)

def build_label_name_lookup():
    # model trả về label 0..79 -> map sang category_id rồi sang tên COCO
    return {lbl: mscoco_category2name.get(cat, str(lbl)) for lbl, cat in mscoco_label2category.items()}

def _measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont):
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        if hasattr(font, "getbbox"):
            bbox = font.getbbox(text)
            return bbox[2] - bbox[0], bbox[3] - bbox[1]
        if hasattr(draw, "textsize"):
            return draw.textsize(text, font=font)
        if hasattr(font, "getsize"):
            return font.getsize(text)
        return max(1, len(text) * 6), 12

def draw_frame(
    frame_bgr,
    labels,
    boxes,
    scores,
    thrh=0.6,
    id2name=None,
    include_names=None,
):
    im_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(im_pil)
    font = ImageFont.load_default()
    W, H = im_pil.size  # clamp toạ độ

    # Chuẩn hoá tensor/list
    if isinstance(labels, (list, tuple)):
        labels = labels[0]
    if isinstance(boxes, (list, tuple)):
        boxes = boxes[0]
    if isinstance(scores, (list, tuple)):
        scores = scores[0]

    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu()
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.detach().cpu()
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu()

    labels = labels.reshape(-1).to(torch.int64)
    boxes = boxes.reshape(-1, 4).to(torch.float32)
    scores = scores.reshape(-1).to(torch.float32)

    keep = scores >= float(thrh)
    names = [id2name.get(int(l), str(int(l))) if id2name else str(int(l)) for l in labels.tolist()]

    if include_names:
        include_set = {n.strip().lower() for n in include_names}
        name_keep = torch.tensor([n.lower() in include_set for n in names], dtype=torch.bool)
        keep = keep & name_keep

    labels = labels[keep]
    boxes = boxes[keep]
    scores = scores[keep]
    names = [n for k, n in zip(keep.tolist(), names) if k]

    for box, name, sc in zip(boxes.tolist(), names, scores.tolist()):
        x1, y1, x2, y2 = box

        # Clamp bbox vào khung ảnh
        x1 = max(0.0, min(float(x1), W - 1))
        y1 = max(0.0, min(float(y1), H - 1))
        x2 = max(0.0, min(float(x2), W - 1))
        y2 = max(0.0, min(float(y2), H - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        # Vẽ bbox
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

        # Vẽ nhãn
        text = f"{name} {sc:.2f}"
        tw, th = _measure_text(draw, text, font)
        bg_x0 = int(x1)
        bg_y1 = int(y1)
        bg_y0 = max(0, bg_y1 - th - 2)
        bg_x1 = min(W - 1, bg_x0 + tw + 4)
        if bg_y1 < bg_y0:
            bg_y0, bg_y1 = bg_y1, bg_y0

        draw.rectangle([bg_x0, bg_y0, bg_x1, bg_y1], fill="red")
        draw.text((bg_x0 + 2, bg_y0), text, fill="white", font=font)

    return cv2.cvtColor(np.array(im_pil), cv2.COLOR_RGB2BGR)

class RTDETRModel(nn.Module):
    def __init__(self, cfg, checkpoint_path, device="cuda"):
        super().__init__()
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
            cfg.model.load_state_dict(state)
        self.model = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()
        self.to(device)
        self.eval()
        self.device = device

    def forward(self, images, orig_target_sizes):
        outputs = self.model(images)
        return self.postprocessor(outputs, orig_target_sizes)

def main(args):
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = YAMLConfig(args.config, resume=args.resume)
    model = RTDETRModel(cfg, args.resume, device)

    transform = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])

    # Mở webcam
    # Windows: CAP_DSHOW có thể giúp mở nhanh hơn
    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW) if os.name == 'nt' else cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index: {args.cam}")

    # Set độ phân giải nếu có
    if args.width > 0 and args.height > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # VideoWriter nếu cần ghi lại
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w_out = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or args.width or 640
        h_out = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or args.height or 480
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 25.0
        writer = cv2.VideoWriter(args.save, fourcc, fps, (w_out, h_out))

    id2name = build_label_name_lookup()
    cls_filter = [c.strip() for c in args.classes] if args.classes else None

    last = time.perf_counter()
    frame_cnt = 0

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_cnt += 1
            im_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            w, h = im_pil.size
            orig_size = torch.tensor([[w, h]], device=device)
            im_data = transform(im_pil)[None].to(device)

            # autocast theo device
            if device.startswith("cuda") and torch.cuda.is_available():
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    labels, boxes, scores = model(im_data, orig_size)
            else:
                labels, boxes, scores = model(im_data, orig_size)

            frame_out = draw_frame(
                frame,
                labels,
                boxes,
                scores,
                thrh=args.threshold,
                id2name=id2name,
                include_names=cls_filter,
            )

            # FPS
            now = time.perf_counter()
            if now - last >= 1.0:
                fps_txt = f"FPS: {frame_cnt/(now-last):.1f}"
                frame_cnt = 0
                last = now
            else:
                fps_txt = ""
            if fps_txt:
                cv2.putText(frame_out, fps_txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow("RT-DETR Webcam", frame_out)
            if writer:
                writer.write(frame_out)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):  # ESC or q
                break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RT-DETR webcam inference with class names")
    parser.add_argument("-c", "--config", type=str, required=True, help="Config file path")
    parser.add_argument("-r", "--resume", type=str, required=True, help="Checkpoint path")
    parser.add_argument("-d", "--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--cam", type=int, default=0, help="Camera index (0 là webcam mặc định)")
    parser.add_argument("--width", type=int, default=0, help="Capture width (0: giữ mặc định)")
    parser.add_argument("--height", type=int, default=0, help="Capture height (0: giữ mặc định)")
    parser.add_argument("-t", "--threshold", type=float, default=0.6, help="Score threshold")
    parser.add_argument("--classes", nargs="*", default=None, help="Chỉ hiển thị các lớp này (vd: person car dog)")
    parser.add_argument("--save", type=str, default="", help="Ghi video ra file mp4 (bỏ trống để không ghi)")
    args = parser.parse_args()
    main(args)