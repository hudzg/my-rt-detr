# RT-DETR PyTorch

Repo này là code của **RT-DETR (Real-Time Detection Transformer)** bằng **PyTorch**.

---

## 🚀 Cài đặt & Thiết lập môi trường

### 1️⃣ Di chuyển vào thư mục dự án

```bash
cd rtdetr_pytorch
```

### 2️⃣ Tạo môi trường ảo (Python 3.11)

```bash
uv venv --python 3.11
```

### 3️⃣ Kích hoạt môi trường ảo

* **Windows:**

```bash
.venv\Scripts\activate
```

* **Linux/macOS:**

```bash
source .venv/bin/activate
```

### 4️⃣ Cài đặt các gói phụ thuộc

```bash
uv pip install -r requirements.txt
```

---

## 🧠 Ví dụ chạy mô hình

### 🖼️ Dự đoán trên ảnh (ResNet-50 backbone)

```bash
uv run tools/infer.py \
  -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
  -r rtdetr_r50vd_6x_coco_from_paddle.pth \
  -f 000000000139.jpg \
  -d cuda
```

### 🏷️ Dự đoán trên ảnh (hiển thị nhãn)

```bash
uv run tools/infer_with_labels.py \
  -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
  -r rtdetr_r50vd_6x_coco_from_paddle.pth \
  -f 000000000139.jpg \
  -d cuda
```

### 🎥 Dự đoán trên video (hiển thị nhãn)

```bash
uv run tools/infer_video_with_labels.py \
  -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
  -r rtdetr_r50vd_6x_coco_from_paddle.pth \
  -i car-detection.mp4 \
  -o output.mp4 \
  -d cuda
```

### 📸 Dự đoán trực tiếp qua webcam (ResNet-18 backbone)

```bash
uv run tools/infer_webcam_with_labels.py \
  -c configs/rtdetr/rtdetr_r18vd_6x_coco.yml \
  -r rtdetr_r18vd_dec3_6x_coco_from_paddle.pth \
  --cam 0
```
