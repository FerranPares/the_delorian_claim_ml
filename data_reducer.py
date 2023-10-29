import json
from pathlib import Path

def reduce_annotations(annotations_dict: dict) -> dict:
    new_images = [image_json for image_json in annotations_dict['images'] if "0.jpg" in image_json["file_name"]]
    annotations_dict['images'] = new_images
    return annotations_dict


def read_json(json_path: Path) -> dict:
    with open(json_path) as fp:
        _dict = json.load(fp)
    return _dict

def write_json(json_dict: dict, json_path: Path):
    with open(json_path, "w") as fp:
        json.dump(json_dict, fp)


ANNOTATIONS_PATH = Path(__file__).parent / "data" / "CarDD_COCO" / "annotations"
SMALL_ANNOTATIONS_PATH = Path(__file__).parent / "small_data" / "CarDD_COCO" / "annotations"

def main():
    annotations_filenames_list = [
        "instances_train2017.json",
        "instances_val2017.json",
        "instances_test2017.json",
    ]
    for annotation_filename in annotations_filenames_list:
        annotations = read_json(ANNOTATIONS_PATH / annotation_filename)
        reduced_annotations = reduce_annotations(annotations)
        write_json(reduced_annotations, SMALL_ANNOTATIONS_PATH / annotation_filename)


if __name__ == '__main__':
    main()
