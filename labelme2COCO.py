# --------------------------------------------------------
# Written by JeongMin Kim(jm.kim@dankook.ac.kr)
#
# labelme2COCO.py
#
# reference from https://github.com/labelmeai/labelme/blob/main/examples/instance_segmentation/labelme2coco.py
# ----------------------------------------------------
import argparse
import os
import os.path as osp
import shutil
from tqdm import tqdm

from COCOformat import ImageInfo, Annotation, KeypointDB

import json
import glob
import labelme
import imgviz


def arg_parser():
    parser = argparse.ArgumentParser(description="labelme2COCO")
    parser.add_argument("--input_dir",
                        default='./data/coco/foot/',
                        help="input annotated your directory")
    parser.add_argument("--output_dir",
                        default='./data/coco/test/',
                        help="output dataset directory"
                        )
    parser.add_argument("--labels",
                        help="labels file",
                        required=False,
                        )

    parser.add_argument("--imgaug",
                        help="If you want to augment, This flag is True",
                        required=True,
                        )
    parser.add_argument("--category",
                        help="category name (e.g. hand, foot)",
                        required=True)
    args = parser.parse_args()

    return args


class Labelme2COCOKeypointDB(KeypointDB):
    def __init__(self, args, json_file=None, is_load_coco=False):
        super().__init__(args, json_file=json_file, is_load_coco=False)
        self.output_dir = args.output_dir
        self.input_dir = args.input_dir
        self.json_file_list = json_file

        if not is_load_coco:
            if self.json_file_list is None:
                raise Exception("Please Input Json file list")

            else:
                self._init_categories()
                self.generate_db()
        else:
            # _load_coco_json
            # Only One json_file
            self.load_coco_json()
            pass

    def _init_categories(self):
        """ Read First annotation Json file and apply base information for categories """
        if self.args.category == "foot":
            categories = dict(
                keypoints=[
                    "1MP", "1PP", "1CiB",
                    "2MP", "2PP", "2CiB",
                    "3MP", "3PP", "3CiB",
                    "4MP", "4PP",
                    "5MP", "5PP",
                    "UCB", "UNB", "DNB", "DCB"
                ],
                skeletons = [[1,2],[2,3],[3,15],
                             [4,5],[5,6],[6,15],
                             [7,8],[8,16],
                             [10,11],[11,9],[9,17],
                             [12,13],[13,14],[14,17],
                             [15,16],[17,16]],
                id=1,
                name="foot",
                supercategory="foot"
            )
        elif self.args.category == "hand":
            categories = dict(
                keypoints=[
                    "1MP", "1PP", "1CiB",
                    "2MP", "2PP", "2CiB",
                    "3MP", "3PP", "3CiB",
                    "4MP", "4PP",
                    "5MP", "5PP",
                    "UCB", "UNB", "DNB", "DCB"
                ],
                skeletons = [[1,2],[2,3],[3,15],
                             [4,5],[5,6],[6,15],
                             [7,8],[8,16],
                             [10,11],[11,9],[9,17],
                             [12,13],[13,14],[14,17],
                             [15,16],[17,16]],
                id=1,
                name="hand",
                supercategory="hand"
            )
        else:
            raise ValueError(f"지원하지 않는 category입니다: {self.args.category}")

        # shapes에서 추가 keypoints가 있으면 반영
        shapes = labelme.LabelFile(filename=self.json_file_list[0]).shapes
        for s in shapes:
            if s['label'].lower() == self.args.category or s['label'].lower() == "letter":
                continue
            if s["label"] not in categories['keypoints']:
                categories['keypoints'].append(s["label"])

        self.db['categories'].append(categories)
        print(f"Initialize Basic categories for {self.args.category}")
        return
    
    def generate_db(self):
        print("Generating Keypoint DB")
        print("=============================================")

        keypoint_order = [
            "1MP", "1PP", "1CiB",
            "2MP", "2PP", "2CiB",
            "3MP", "3PP", "3CiB",
            "4MP", "4PP",
            "5MP", "5PP",
            "UCB", "UNB", "DNB", "DCB"
        ]

        for image_id, filename in tqdm(enumerate(self.json_file_list), total=len(self.json_file_list), desc="Converting JSONs"):
            print("Processing:", filename)

            base = osp.splitext(osp.basename(filename))[0]
            with open(filename, 'r') as file:
                data = json.load(file)

                # 이미지 파일 찾기
                origin_image_paths = glob.glob(osp.join(self.input_dir, "**", data.get("imagePath")), recursive=True)
                if not origin_image_paths:
                    raise FileNotFoundError(f"Image file not found: {data.get('imagePath')} in {self.input_dir}")
                origin_image_path = origin_image_paths[0] 

                # 이미지 복사
                out_img_file = osp.join(self.output_dir, "JPEGImages", base + ".jpg")
                shutil.copy(src=origin_image_path, dst=out_img_file)

                # COCO Image 정보 생성
                image_info = ImageInfo(
                    license=0,
                    coco_url='',
                    flickr_url='',
                    file_name=osp.relpath(out_img_file, osp.dirname(out_img_file)),
                    height=data.get("imageHeight"),
                    width=data.get("imageWidth"),
                    date_captured=None,
                    id=image_id,
                )

                # Keypoints 및 Bounding Box 처리
                annotation_db = Annotation(
                    area=0,
                    iscrowd=0,
                    image_id=image_id,
                    bbox=[],
                    category_id=1,
                    id=image_id,
                    keypoints=[],
                    num_keypoints=17,
                )

                # Keypoints 매핑을 위한 딕셔너리
                keypoints_dict = {kp: [0, 0, 0] for kp in keypoint_order}  # (x, y, visibility)

                num_keypoints = 0
                for s in data['shapes']:
                    label = s.get("label")
                    if label == "letter":
                        continue  # letter는 keypoint로 절대 안 들어가게 예외 처리
                    if label in keypoints_dict:
                        x, y = s.get("points")[0]
                        keypoints_dict[label] = [x, y, 2]  # 2: Visible
                        num_keypoints += 1
                    elif s["shape_type"] == "rectangle":
                        x0, y0 = s.get("points")[0]
                        x1, y1 = s.get("points")[1]
                        annotation_db.bbox = [x0, y0, x1 - x0, y1 - y0] # x, y, w, h format
                        annotation_db.area = annotation_db.bbox[2] * annotation_db.bbox[3]

                annotation_db.num_keypoints = num_keypoints
                annotation_db.keypoints = [coord for kp in keypoint_order for coord in keypoints_dict[kp]]  # 순서 맞춰 변환

                self.db['images'].append(image_info)
                self.db['annotations'].append(annotation_db)

        print("✅ Keypoints 변환 완료!")

    def augmentation(self):
        print("Augmentation DB")
        print("=============================================")


def main():
    args = arg_parser()

    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not osp.exists(osp.join(args.output_dir, "JPEGImages")):
        os.makedirs(osp.join(args.output_dir, "JPEGImages"))

    print("Creating Dataset to: ", args.output_dir)
    print("=============================================")

    json_files = glob.glob(osp.join(args.input_dir, "**", "*.json"), recursive=True)
    print(f"총 {len(json_files)}개의 json 파일을 찾았습니다.")

    db = Labelme2COCOKeypointDB(args, json_file=json_files)
    db.saver()


if __name__ == '__main__':
    main() 