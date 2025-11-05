import torch
import torch.nn as nn
import torchvision.transforms as T
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.core import YAMLConfig

# -----------------------
# Vẽ bounding box
# -----------------------
def draw_frame(frame, labels, boxes, scores, thrh=0.6, class_names=None):
    im_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(im_pil)

    keep = scores[0] > thrh
    boxes = boxes[0][keep]
    labels = labels[0][keep]
    scores = scores[0][keep]

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.tolist()
        cls_id = int(labels[i].item())
        score = float(scores[i].item())

        name = str(cls_id) if class_names is None else class_names[cls_id]
        text = f"{name} {score:.2f}"

        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1), text, fill="blue", font=ImageFont.load_default())

    return cv2.cvtColor(np.array(im_pil), cv2.COLOR_RGB2BGR)


# -----------------------
# Model wrapper
# -----------------------
class RTDETRModel(nn.Module):
    def __init__(self, cfg, checkpoint_path, device="cuda"):
        super().__init__()
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
            cfg.model.load_state_dict(state)
        self.model = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()
        self.device = device
        self.to(device)

    def forward(self, images, orig_target_sizes):
        outputs = self.model(images)
        return self.postprocessor(outputs, orig_target_sizes)


# -----------------------
# Main
# -----------------------
def main(args):
    cfg = YAMLConfig(args.config, resume=args.resume)
    model = RTDETRModel(cfg, args.resume, args.device)

    transform = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])

    # Mở video input
    cap = cv2.VideoCapture(args.input)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # VideoWriter để lưu kết quả
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Convert frame sang tensor
        im_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        w, h = im_pil.size
        orig_size = torch.tensor([[w, h]]).to(args.device)
        im_data = transform(im_pil)[None].to(args.device)

        with torch.cuda.amp.autocast():
            labels, boxes, scores = model(im_data, orig_size)

        # Vẽ kết quả
        frame_out = draw_frame(frame, labels.cpu(), boxes.cpu(), scores.cpu(), thrh=args.threshold)
        out.write(frame_out)

        if frame_idx % 10 == 0:
            print(f"Processed frame {frame_idx}")

    cap.release()
    out.release()
    print(f"Done. Video saved at {args.output}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True, help="Config file path")
    parser.add_argument("-r", "--resume", type=str, required=True, help="Checkpoint path")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input video path")
    parser.add_argument("-o", "--output", type=str, default="result.mp4", help="Output video path")
    parser.add_argument("-d", "--device", type=str, default="cuda")
    parser.add_argument("-t", "--threshold", type=float, default=0.6)

    args = parser.parse_args()
    main(args)