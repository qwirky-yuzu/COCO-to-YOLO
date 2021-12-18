import os
from json import JSONDecodeError
from os import path
import cv2
import json
import pathlib


class ConvertCOCOToYOLO:
    """
    Takes in the path to COCO annotations and outputs YOLO annotations in multiple .txt files.
    COCO annotation are to be JSON formart as follows:

        "annotations":{
            "area":2304645,
            "id":1,
            "image_id":10,
            "category_id":4,
            "bbox":[
                0::704
                1:620
                2:1401
                3:1645
            ]
        }
        
    """

    def __init__(self, img_folder, json_path):
        self.img_folder = img_folder
        self.json_path = json_path

    def check_path_variables(self):

        if self.img_folder is None or self.json_path is None:
            return False
        if path.exists(self.img_folder) and path.exists(self.json_path):
            if self.get_img_shape(os.path.join(self.img_folder, os.listdir(self.img_folder)[0])) is not None:
                try:
                    json.load(open(self.json_path))
                    return True
                except JSONDecodeError:
                    print("cant't open Json File")
                    return False
            else:
                return False
        else:
            print("Path to images or json file are not correctly defined")
            return False

    def get_img_shape(self, img_path):
        img = cv2.imread(img_path)

        try:
            return img.shape
        except AttributeError:
            print('error, no image found at', img_path)
            return None

    def convert_labels(self, img_path, x1, y1, x2, y2):
        """
        Definition: Parses label files to extract label and bounding box
        coordinates. Converts (x1, y1, x1, y2) KITTI format to
        (x, y, width, height) normalized YOLO format.
        """

        def sorting(l1, l2):
            if l1 > l2:
                lmax, lmin = l1, l2
                return lmax, lmin
            else:
                lmax, lmin = l2, l1
                return lmax, lmin

        size = self.get_img_shape(img_path)
        xmax, xmin = sorting(x1, x2)
        ymax, ymin = sorting(y1, y2)
        dw = 1. / size[1]
        dh = 1. / size[0]
        x = (xmin + xmax) / 2.0
        y = (ymin + ymax) / 2.0
        w = xmax - xmin
        h = ymax - ymin
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return (x, y, w, h)

    def convert(self, annotation_key='annotations', img_id='image_id', cat_id='category_id', bbox='bbox'):

        if not self.check_path_variables():
            return None

        # Enter directory to read JSON file
        data = json.load(open(self.json_path))

        check_set = set()

        # Retrieve data
        for i in range(len(data[annotation_key])):

            # Get required data
            image_id = f'{data[annotation_key][i][img_id]}'
            category_id = f'{data[annotation_key][i][cat_id]}'
            bbox = data[annotation_key][i]["bbox"]

            # Retrieve Image Name
            image_name = ""
            images_metadata = data["images"]
            for image_object in images_metadata:
                if image_object["id"] == int(image_id):
                    image_name = image_object["file_name"]

            image_path = os.path.join(self.img_folder, image_name)
            # Convert the data
            kitti_bbox = [bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1]]
            yolo_bbox = self.convert_labels(image_path, kitti_bbox[0], kitti_bbox[1], kitti_bbox[2], kitti_bbox[3])

            # Prepare for export
            path_to_dataset = pathlib.Path(self.img_folder).parent.resolve()
            filename = f'{image_name}.txt'
            content = f"{category_id} {yolo_bbox[0]} {yolo_bbox[1]} {yolo_bbox[2]} {yolo_bbox[3]}"
            path_to_file = os.path.join(path_to_dataset, "labels", filename)
            # Export 
            if image_id in check_set:
                # Append to existing file as there can be more than one label in each image
                file = open(path_to_file, "a")
                file.write("\n")
                file.write(content)
                file.close()

            elif image_id not in check_set:
                check_set.add(image_id)
                # Write files
                file = open(path_to_file, "w")
                file.write(content)
                file.close()


current_path = pathlib.Path(__file__).parent.resolve()
# To run in as a class
ConvertCOCOToYOLO(img_folder="dataset/images", json_path='dataset/test.json').convert()
