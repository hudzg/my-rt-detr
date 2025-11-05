import torch
import cv2
from src.core import build_model
from src.core import post_process

# Load config và model
config = "configs/rtdetr/rtdetr_r18vd_6x_coco.yml"
checkpoint = "rtdetr_r18vd_dec3_6x_coco_from_paddle.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model(config)
model.load_state_dict(torch.load(checkpoint, map_location=device))
model.eval().to(device)

# Load video
cap = cv2.VideoCapture("input.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output.mp4", fourcc, cap.get(cv2.CAP_PROP_FPS),
                      (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img).permute(2,0,1).float().unsqueeze(0).to(device) / 255.0

    # Forward
    with torch.no_grad():
        outputs = model(img_tensor)

    # Post-process (tùy vào repo, có thể đã có sẵn hàm decode)
    boxes, scores, labels = post_process(outputs, conf_thres=0.5)

    # Vẽ box
    for (x1, y1, x2, y2), score, label in zip(boxes, scores, labels):
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
        cv2.putText(frame, f"{label}:{score:.2f}", (int(x1), int(y1)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()
