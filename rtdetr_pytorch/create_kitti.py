import os
import glob

def create_kitti_filelists(root_dir, output_dir=None):
    if output_dir is None:
        output_dir = root_dir

    # Cấu hình các thư mục con theo chuẩn Ultralytics/YOLO
    # Giả định cấu trúc: root/images/train, root/labels/train
    sub_dirs = ["train", "val"]
    
    # Danh sách class của KITTI (theo thứ tự index 0-7 mà bạn cung cấp trước đó)
    label_list = [
        "car", "van", "truck", "pedestrian", 
        "person_sitting", "cyclist", "tram", "misc"
    ]

    # 1. Tạo file label_list.txt
    label_path = os.path.join(output_dir, "label_list.txt")
    with open(label_path, "w") as f:
        for lab in label_list:
            f.write(lab + "\n")
    print(f"✅ Đã tạo {label_path} ({len(label_list)} lớp)")

    # 2. Tạo file train.txt và val.txt
    for split in sub_dirs:
        img_dir = os.path.join(root_dir, "images", split)
        lbl_dir = os.path.join(root_dir, "labels", split)

        if not os.path.exists(img_dir):
            print(f"⚠️ Không tìm thấy thư mục {img_dir}, bỏ qua {split}.")
            continue

        output_file = os.path.join(output_dir, f"{split}.txt")
        
        # Tìm tất cả ảnh (hỗ trợ jpg, png, jpeg)
        image_files = []
        for ext in ['*.jpg', '*.png', '*.jpeg']:
            image_files.extend(glob.glob(os.path.join(img_dir, ext)))
            
        with open(output_file, "w") as f:
            count = 0
            for img_path_abs in image_files:
                # Lấy tên file không đuôi (vd: 000002)
                file_id = os.path.splitext(os.path.basename(img_path_abs))[0]
                
                # Tạo đường dẫn label tương ứng
                lbl_path_abs = os.path.join(lbl_dir, file_id + ".txt")
                
                # Chỉ ghi nếu file label tồn tại
                if os.path.exists(lbl_path_abs):
                    # Chuyển đổi sang đường dẫn tương đối (để code portable hơn)
                    # VD: images/train/000002.png labels/train/000002.txt
                    rel_img = os.path.relpath(img_path_abs, root_dir)
                    rel_lbl = os.path.relpath(lbl_path_abs, root_dir)
                    
                    f.write(f"{rel_img} {rel_lbl}\n")
                    count += 1
        
        print(f"✅ Đã tạo {output_file} ({count} cặp ảnh-nhãn)")

if __name__ == "__main__":
    # Thay đổi đường dẫn này trỏ đến thư mục dataset/kitti của bạn
    target_dir = "dataset/kitti" 

    if os.path.exists(target_dir):
        create_kitti_filelists(target_dir)
    else:
        print(f"❌ Lỗi: Không tìm thấy thư mục {target_dir}")