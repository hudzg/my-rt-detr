"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
Adapted for Cityscapes with VOC-style XML labels.
"""

import torch
import torchvision
import torchvision.transforms.functional as TVF 
import os
from PIL import Image
from typing import Optional, Callable, Dict, Any

try:
    from defusedxml.ElementTree import parse as ET_parse
except ImportError:
    from xml.etree.ElementTree import parse as ET_parse

# Giả định file này nằm trong cấu trúc thư mục tương tự code gốc của bạn
# Nếu chạy độc lập, bạn có thể comment dòng import DetDataset và thay bằng object
from ._dataset import DetDataset 
from .._misc import convert_to_tv_tensor
from ...core import register

@register
class CityscapesDetection(DetDataset):
    __inject__ = ['transforms', ]

    def __init__(self, 
                 root: str, 
                 ann_file: str = "train.txt", 
                 label_file: str = "label_list.txt", 
                 transforms: Optional[Callable] = None):
        """
        Args:
            root (str): Đường dẫn gốc tới dataset (vd: dataset/cityscapes)
            ann_file (str): Tên file list (vd: train.txt hoặc val.txt)
            label_file (str): Tên file chứa danh sách class (label_list.txt)
            transforms (Callable): Hàm transform ảnh
        """
        # super().__init__(root, transforms, None, None)
        
        # 1. Đọc file list (chứa cặp: đường_dẫn_ảnh đường_dẫn_xml)
        ann_path = os.path.join(root, ann_file)
        if not os.path.exists(ann_path):
            raise FileNotFoundError(f"Không tìm thấy file annotation list: {ann_path}")

        with open(ann_path, 'r') as f:
            lines = [x.strip() for x in f.readlines() if x.strip()]
            lines = [x.split(' ') for x in lines]

        # Lưu đường dẫn tuyệt đối hoặc nối với root
        self.images = [os.path.join(root, lin[0]) for lin in lines]
        self.xml_paths = [os.path.join(root, lin[1]) for lin in lines]
        
        assert len(self.images) == len(self.xml_paths), "Số lượng ảnh và file XML không khớp!"

        # 2. Đọc label list để map tên class sang ID
        label_path = os.path.join(root, label_file)
        if not os.path.exists(label_path):
             raise FileNotFoundError(f"Không tìm thấy file label list: {label_path}")

        with open(label_path, 'r') as f:
            labels = f.readlines()
            labels = [lab.strip() for lab in labels if lab.strip()]

        self.transforms = transforms
        # Tạo map: {'person': 0, 'car': 1, ...}
        self.labels_map = {lab: i for i, lab in enumerate(labels)}
        
        # Lưu lại danh sách class để dùng khi cần visual
        self.classes = labels 

    def __getitem__(self, index: int):
        image, target = self.load_item(index)
        if self.transforms is not None:
            image, target, _ = self.transforms(image, target, self)        
        return image, target

    def load_item(self, index: int):
        # 1. Load Ảnh
        img_path = self.images[index]
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
            
        image = Image.open(img_path).convert("RGB")
        
        # 2. Load và Parse XML
        xml_path = self.xml_paths[index]
        if not os.path.exists(xml_path):
             raise FileNotFoundError(f"XML not found: {xml_path}")
             
        # Parse XML thành dictionary
        target_dict = self.parse_voc_xml(ET_parse(xml_path).getroot())
        
        # 3. Chuẩn hóa dữ liệu đầu ra
        output = {}
        output["image_id"] = torch.tensor([index])
        
        # Khởi tạo list rỗng
        for k in ['area', 'boxes', 'labels', 'iscrowd']:
            output[k] = []
            
        # Lấy danh sách object từ kết quả parse
        # Lưu ý: file XML VOC chuẩn sẽ có cấu trúc annotation -> object
        ann = target_dict.get('annotation', target_dict)
        objects = ann.get('object', [])
        if not isinstance(objects, list):
            objects = [objects] # Xử lý trường hợp chỉ có 1 object

        for blob in objects:
            # Lọc bỏ class không có trong label_list (nếu có)
            if blob['name'] not in self.labels_map:
                continue
                
            # Lấy bbox: xmin, ymin, xmax, ymax
            bndbox = blob['bndbox']
            box = [float(bndbox['xmin']), float(bndbox['ymin']), 
                   float(bndbox['xmax']), float(bndbox['ymax'])]
            
            output["boxes"].append(box)
            output["labels"].append(blob['name'])
            # Diện tích = (xmax - xmin) * (ymax - ymin)
            output["area"].append((box[2] - box[0]) * (box[3] - box[1]))
            output["iscrowd"].append(0) # Cityscapes XML convert thường không set difficult/crowd

        w, h = image.size
        
        # Chuyển sang Tensor
        if len(output["boxes"]) > 0:
            boxes = torch.tensor(output["boxes"])
        else:
            boxes = torch.zeros(0, 4) # Xử lý ảnh không có object
            
        output['boxes'] = convert_to_tv_tensor(boxes, 'boxes', box_format='xyxy', spatial_size=[h, w])
        output['labels'] = torch.tensor([self.labels_map[lab] for lab in output["labels"]], dtype=torch.int64)
        output['area'] = torch.tensor(output['area'])
        output["iscrowd"] = torch.tensor(output["iscrowd"])
        output["orig_size"] = torch.tensor([w, h])
        
        return image, output

    def parse_voc_xml(self, node: Any) -> Dict[str, Any]:
        """
        Hàm đệ quy để parse XML ElementTree thành Dictionary.
        Giống hệt logic của torchvision.datasets.VOCDetection
        """
        voc_dict = {}
        children = list(node)
        if children:
            def_dic = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            if node.tag == "annotation":
                def_dic["object"] = [val for val in def_dic["object"]]
            voc_dict = {
                node.tag: {
                    ind: v[0] if len(v) == 1 else v for ind, v in def_dic.items()
                }
            }
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict

    def __len__(self) -> int:
        return len(self.images)

# Helper import for recursive parsing
import collections