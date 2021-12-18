import os
from json import JSONDecodeError
from os import path
import json
import pathlib


class ConvertCOCOToYOLO:
    """
    Takes in the path to COCO annotations and optional the path, where the YOLO annotations shall be saved
    in multiple .txt files.
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

    def __init__(self, json_path, label_path):
        self.json_path = json_path
        self.label_path = label_path

    def check_paths(self):
        if self.label_path is None or not path.exists(self.label_path):
            self.label_path = pathlib.Path(__file__).parent.resolve()
        if self.json_path is None:
            return False
        if path.exists(self.json_path):
            try:
                json.load(open(self.json_path))
                return True
            except JSONDecodeError:
                print("cant't open Json File")
                return False
        else:
            return False

    def convert_labels(self, x1, y1, x2, y2, size):
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

        if not self.check_paths():
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

            # Retrieve Image Name and width and height
            image_name = ""
            image_height = 0
            image_width = 0
            images_metadata = data["images"]
            for image_object in images_metadata:
                if image_object["id"] == int(image_id):
                    image_name = image_object["file_name"]
                    image_width = image_object["width"]
                    image_height = image_object["height"]

            # Convert the data
            kitti_bbox = [bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1]]
            yolo_bbox = self.convert_labels(kitti_bbox[0], kitti_bbox[1], kitti_bbox[2], kitti_bbox[3],
                                            (image_height, image_width))

            # Prepare for export
            filename = f'{image_name[:-4]}.txt'
            content = f"{category_id} {yolo_bbox[0]} {yolo_bbox[1]} {yolo_bbox[2]} {yolo_bbox[3]}"
            path_to_file = os.path.join(self.label_path, filename)
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


if __name__ == "__main__":
    # To run in as a class
    ConvertCOCOToYOLO(json_path='dataset/test.json', label_path="dataset/labels").convert()
