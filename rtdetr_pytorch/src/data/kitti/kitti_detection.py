import torch
import torchvision
import os
from PIL import Image
from typing import Optional, Callable

from ._dataset import DetDataset
from .._misc import convert_to_tv_tensor
from ...core import register

@register
class KittiDetection(DetDataset):
    __inject__ = ['transforms', ]

    def __init__(self, root: str, ann_file: str = "train.txt", label_file: str = "label_list.txt", transforms: Optional[Callable] = None):
        super().__init__() # Khởi tạo lớp cha nếu cần
        self.root = root
        
        # Đọc file list (đường dẫn ảnh và label)
        file_list_path = os.path.join(root, ann_file)
        if not os.path.exists(file_list_path):
             raise FileNotFoundError(f"Không tìm thấy file danh sách: {file_list_path}")

        with open(file_list_path, 'r') as f:
            lines = [x.strip().split(' ') for x in f.readlines() if x.strip()]
        
        # Tạo đường dẫn tuyệt đối
        self.images = [os.path.join(root, lin[0]) for lin in lines]
        self.targets = [os.path.join(root, lin[1]) for lin in lines]
        
        assert len(self.images) == len(self.targets), "Số lượng ảnh và label không khớp!"

        # Đọc danh sách label (để map tên nếu cần, hoặc để kiểm tra số class)
        with open(os.path.join(root, label_file), 'r') as f:
            self.classes = [x.strip() for x in f.readlines()]

        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int):
        image, target = self.load_item(index)
        if self.transforms is not None:
            image, target, _ = self.transforms(image, target, self)        
        return image, target

    def load_item(self, index: int):
        # 1. Đọc ảnh
        image_path = self.images[index]
        # Dùng .convert("RGB") để đảm bảo ảnh luôn có 3 kênh (xử lý lỗi ảnh đen trắng hoặc RGBA)
        image = Image.open(image_path).convert("RGB")
        w, h = image.size

        # 2. Đọc file Label (.txt format YOLO)
        label_path = self.targets[index]
        
        boxes = []
        labels = []
        area = []
        iscrowd = []

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5: continue # Bỏ qua dòng lỗi
                
                # Format YOLO: class_id x_center y_center w h (normalized 0-1)
                cls_id = int(parts[0])
                cx = float(parts[1])
                cy = float(parts[2])
                bw = float(parts[3])
                bh = float(parts[4])

                # Chuyển đổi YOLO (x_c, y_c, w, h) -> XYXY (absolute pixels)
                # Đây là bước quan trọng nhất để khớp với format của DetDataset
                x_c = cx * w
                y_c = cy * h
                half_w = (bw * w) / 2
                half_h = (bh * h) / 2

                x1 = x_c - half_w
                y1 = y_c - half_h
                x2 = x_c + half_w
                y2 = y_c + half_h
                
                # Clip box nằm trong ảnh
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                boxes.append([x1, y1, x2, y2])
                labels.append(cls_id)
                area.append((x2 - x1) * (y2 - y1))
                iscrowd.append(0)

        # 3. Đóng gói vào dictionary chuẩn Output
        output = {}
        output["image_id"] = torch.tensor([index])
        
        # Xử lý trường hợp ảnh không có object nào (background image)
        if len(boxes) > 0:
            raw_boxes = torch.tensor(boxes, dtype=torch.float32)
            output['labels'] = torch.tensor(labels, dtype=torch.int64)
            output['area'] = torch.tensor(area, dtype=torch.float32)
            output["iscrowd"] = torch.tensor(iscrowd, dtype=torch.int64)
        else:
            raw_boxes = torch.zeros((0, 4), dtype=torch.float32)
            output['labels'] = torch.zeros((0,), dtype=torch.int64)
            output['area'] = torch.zeros((0,), dtype=torch.float32)
            output["iscrowd"] = torch.zeros((0,), dtype=torch.int64)

        # convert_to_tv_tensor là hàm tiện ích của framework bạn đang dùng
        # Nó chuẩn hóa format box về đúng dạng model yêu cầu (thường là cxcywh hoặc xyxy normalized)
        output['boxes'] = convert_to_tv_tensor(raw_boxes, 'boxes', box_format='xyxy', spatial_size=[h, w])
        output["orig_size"] = torch.tensor([w, h])
        
        return image, output