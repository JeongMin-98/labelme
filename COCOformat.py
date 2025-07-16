from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import json

@dataclass
class ImageInfo:
    license: int
    file_name: str
    height: int
    width: int
    id: int
    coco_url: str = ''
    flickr_url: str = ''
    date_captured: str = None

@dataclass
class Annotation:
    area: float
    iscrowd: int
    image_id: int
    bbox: list
    category_id: int
    id: int
    keypoints: list
    num_keypoints: int

class KeypointDB:
    def __init__(self, args, json_file=None, is_load_coco=False):
        self.args = args
        self.db = {
            'images': [],
            'annotations': [],
            'categories': []
        }
        self.output_dir = getattr(args, 'output_dir', None)
        self.input_dir = getattr(args, 'input_dir', None)
        self.json_file_list = json_file
        if not is_load_coco:
            self._init_categories()
            self.generate_db()
        else:
            self.load_coco_json()

    def _init_categories(self):
        # 상속받는 쪽에서 구현
        pass

    def generate_db(self):
        # 상속받는 쪽에서 구현
        pass

    def load_coco_json(self):
        # COCO json 불러오기용 (필요시 구현)
        pass

    def saver(self):
        # COCO json 저장
        output_path = self.output_dir if self.output_dir else '.'
        with open(f'{output_path}/annotations.json', 'w', encoding='utf-8') as f:
            json.dump(self._to_serializable(), f, ensure_ascii=False, indent=2)
        print(f"Saved COCO-format json to {output_path}/annotations.json")

    def _to_serializable(self):
        # dataclass 객체를 dict로 변환
        return {
            'images': [asdict(img) for img in self.db['images']],
            'annotations': [asdict(ann) for ann in self.db['annotations']],
            'categories': self.db['categories']
        } 