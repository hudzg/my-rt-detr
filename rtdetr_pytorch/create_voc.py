import os
import xml.etree.ElementTree as ET

def create_voc_filelists(root_dir, output_dir=None):
    if output_dir is None:
        output_dir = root_dir

    # Đường dẫn chuẩn
    img_dir = os.path.join(root_dir, "JPEGImages")
    ann_dir = os.path.join(root_dir, "Annotations")
    imagesets_main = os.path.join(root_dir, "ImageSets", "Main")

    # -------------------
    # 1. Tạo file trainval.txt
    print("Đang xử lý trainval...")
    trainval_file = os.path.join(imagesets_main, "trainval.txt")
    if os.path.exists(trainval_file):
        with open(trainval_file, "r") as f:
            trainval_ids = [x.strip() for x in f.readlines()]
    else:
        trainval_ids = [os.path.splitext(f)[0] for f in os.listdir(ann_dir) if f.endswith(".xml")]

    trainval_path = os.path.join(output_dir, "trainval.txt")
    with open(trainval_path, "w") as f:
        for img_id in trainval_ids:
            img_path = f"JPEGImages/{img_id}.jpg"
            ann_path = f"Annotations/{img_id}.xml"
            f.write(f"{img_path} {ann_path}\n")
    print(f"✅ Đã tạo {trainval_path} ({len(trainval_ids)} dòng)")

    # -------------------
    # 2. Tạo file test.txt
    print("Đang xử lý test...")
    test_file = os.path.join(imagesets_main, "test.txt")
    if os.path.exists(test_file):
        with open(test_file, "r") as f:
            test_ids = [x.strip() for x in f.readlines()]
    else:
        all_ids = [os.path.splitext(f)[0] for f in os.listdir(ann_dir) if f.endswith(".xml")]
        test_ids = [x for x in all_ids if x not in trainval_ids]

    test_path = os.path.join(output_dir, "test.txt")
    with open(test_path, "w") as f:
        for img_id in test_ids:
            img_path = f"JPEGImages/{img_id}.jpg"
            ann_path = f"Annotations/{img_id}.xml"
            f.write(f"{img_path} {ann_path}\n")
    print(f"✅ Đã tạo {test_path} ({len(test_ids)} dòng)")

    # -------------------
    # 3. Tạo file label_list.txt
    label_list = [
        "aeroplane", "bicycle", "bird", "boat", "bottle",
        "bus", "car", "cat", "chair", "cow",
        "diningtable", "dog", "horse", "motorbike", "person",
        "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    ]
    label_path = os.path.join(output_dir, "label_list.txt")
    with open(label_path, "w") as f:
        for lab in label_list:
            f.write(lab + "\n")
    print(f"✅ Đã tạo {label_path} ({len(label_list)} lớp)")

# Chạy hàm (trỏ đúng vào thư mục vừa tải ở Bước 2)
if __name__ == "__main__":
    # Lưu ý đường dẫn này phải khớp với nơi bạn tải dataset
    target_dir = "dataset/voc/VOCdevkit/VOC2007"
    
    if os.path.exists(target_dir):
        create_voc_filelists(target_dir)
    else:
        print(f"❌ Lỗi: Không tìm thấy thư mục {target_dir}. Bạn đã tải dataset chưa?")