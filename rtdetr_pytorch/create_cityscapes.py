import os
import glob
from tqdm import tqdm

def create_cityscapes_filelists(root_dir, output_dir=None, target_beta='0.01'):
    """
    root_dir: Thư mục chứa folder 'xml_labels' và 'leftImg8bit_foggyDBF'
    output_dir: Nơi lưu các file txt kết quả
    target_beta: Chọn mức độ sương mù để map với nhãn (0.005, 0.01, hoặc 0.02)
    """
    if output_dir is None:
        output_dir = root_dir

    # Cấu hình tên thư mục dựa trên ảnh bạn cung cấp
    xml_root = os.path.join(root_dir, "xml_labels")
    img_root = os.path.join(root_dir, "images/leftImg8bit_foggyDBF")

    # Kiểm tra tồn tại
    if not os.path.exists(xml_root) or not os.path.exists(img_root):
        print(f"❌ Lỗi: Không tìm thấy thư mục dữ liệu tại {root_dir}")
        print(f"   - Cần có: {xml_root}")
        print(f"   - Cần có: {img_root}")
        return

    # -------------------
    # Hàm con để tạo list cho từng tập (train/val)
    def generate_split_txt(split_name, output_filename):
        print(f"\n📂 Đang xử lý tập: {split_name} (Beta={target_beta})...")
        
        xml_split_dir = os.path.join(xml_root, split_name)
        img_split_dir = os.path.join(img_root, split_name) # Lưu ý: cấu trúc folder ảnh phải tương ứng (train/val)

        if not os.path.exists(xml_split_dir):
            print(f"⚠️ Không thấy thư mục {split_name} trong xml_labels. Bỏ qua.")
            return

        lines = []
        # Duyệt qua các thành phố (aachen, bochum...)
        cities = [d for d in os.listdir(xml_split_dir) if os.path.isdir(os.path.join(xml_split_dir, d))]
        
        for city in tqdm(cities):
            city_xml_dir = os.path.join(xml_split_dir, city)
            
            # Lấy tất cả file xml trong thành phố đó
            xml_files = glob.glob(os.path.join(city_xml_dir, "*.xml"))
            
            for xml_abs_path in xml_files:
                # 1. Xử lý đường dẫn XML (tương đối so với root_dir)
                # Ví dụ: xml_labels/train/aachen/aachen_xxx.xml
                rel_xml_path = os.path.relpath(xml_abs_path, root_dir)

                # 2. Suy luận đường dẫn Ảnh tương ứng
                # Tên XML: aachen_000000_000019_leftImg8bit.xml
                # Tên Ảnh: aachen_000000_000019_leftImg8bit_foggy_beta_0.01.png
                
                xml_filename = os.path.basename(xml_abs_path)
                # Bỏ đuôi .xml -> aachen_000000_000019_leftImg8bit
                base_name = os.path.splitext(xml_filename)[0] 
                
                # Tạo tên file ảnh foggy
                img_filename = f"{base_name}_foggy_beta_{target_beta}.png"
                
                # Đường dẫn ảnh tuyệt đối để kiểm tra tồn tại
                # leftImg8bit_foggyDBF/train/aachen/anh.png
                img_abs_path = os.path.join(img_split_dir, city, img_filename)
                
                if os.path.exists(img_abs_path):
                    # Đường dẫn ảnh tương đối
                    rel_img_path = os.path.relpath(img_abs_path, root_dir)
                    
                    # Thêm vào list: đường_dẫn_ảnh đường_dẫn_nhãn
                    lines.append(f"{rel_img_path} {rel_xml_path}")
                else:
                    # Trường hợp không tìm thấy ảnh (có thể do chưa giải nén hoặc sai beta)
                    pass

        # Ghi ra file
        out_path = os.path.join(output_dir, output_filename)
        with open(out_path, "w") as f:
            f.write("\n".join(lines))
        
        print(f"✅ Đã tạo {out_path} ({len(lines)} cặp ảnh-nhãn)")

    # -------------------
    # 1. Tạo file train.txt
    generate_split_txt("train", "train.txt")

    # -------------------
    # 2. Tạo file val.txt (Tương đương test.txt trong VOC nếu dùng để đánh giá)
    generate_split_txt("val", "val.txt")

    # -------------------
    # 3. Tạo file label_list.txt (8 lớp detection của Cityscapes)
    # Lưu ý: Thứ tự này phải khớp với file config model của bạn
    label_list = [
        "person",
        "rider",
        "car",
        "truck",
        "bus",
        "train",
        "motorcycle",
        "bicycle"
    ]
    
    label_path = os.path.join(output_dir, "label_list.txt")
    with open(label_path, "w") as f:
        for lab in label_list:
            f.write(lab + "\n")
    print(f"✅ Đã tạo {label_path} ({len(label_list)} lớp)")


if __name__ == "__main__":
    # --- CẤU HÌNH ĐƯỜNG DẪN Ở ĐÂY ---
    # Trỏ vào thư mục cha chứa 'xml_labels' và 'leftImg8bit_foggyDBF'
    # Theo ảnh bạn gửi thì nó là 'dataset/cityscapes'
    DATASET_ROOT = "dataset/cityscapes" 
    
    # Chọn độ mờ sương mù bạn muốn train (0.005, 0.01, 0.02)
    FOG_BETA = "0.01" 

    if os.path.exists(DATASET_ROOT):
        create_cityscapes_filelists(DATASET_ROOT, target_beta=FOG_BETA)
    else:
        print(f"❌ Đường dẫn không đúng: {DATASET_ROOT}")
        print("Vui lòng sửa biến DATASET_ROOT trong code cho đúng vị trí folder của bạn.")