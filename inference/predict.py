from PIL import Image
import numpy as np
from pathlib import Path

from mmdet.apis import DetInferencer
from dataclasses import dataclass

MODEL_CONFIG_PATH = str(Path(__file__).parent / "config" / "DCN_plus_cfg.py")
WEIGHTS_PATH = str(Path(__file__).parent / "weights" / "epoch_24.pth")


# Initialize the DetInferencer
inferencer = DetInferencer(model=MODEL_CONFIG_PATH, weights=WEIGHTS_PATH)
classes = (
    'dent',
    'scratch',
    'crack',
    'glass shatter',
    'lamp broken',
    'tire flat',
)

@dataclass
class Prediction:
    label_id: int
    label_name: str
    score: float
    bbox: list[int, int, int, int]
    visualization: np.ndarray
    
    @classmethod
    def from_mmdet_prediction(cls, mmdet_prediction: dict):
        predictions_list = mmdet_prediction['predictions']
        if len(predictions_list) != 1:
            raise RuntimeError("Unexpected none or multiple predictions!")
        pred = predictions_list[0]
        
        scores_list = pred['scores']
        if not scores_list:
            return {}
        if max(scores_list) != scores_list[0]:
            raise RuntimeError("Something went wrong: First object score is not the highest!")
        
        class_id = pred['labels'][0]
        return cls(
            label_id=class_id,
            label_name=classes[class_id],
            score=pred['scores'][0],
            bbox=pred['bboxes'][0],
            visualization=mmdet_prediction['visualization'][0]
        )

def get_prediction(np_img: np.ndarray) -> dict:
    # Perform inference
    mmdet_prediction_dict = inferencer(np_img, return_vis=True)
    # Post-process and instantiates Prediction object
    return Prediction.from_mmdet_prediction(mmdet_prediction_dict)



if __name__ == '__main__':
    img_path = Path("data/CarDD_COCO/test2017/000040.jpg")
    img = Image.open(img_path)
    np_img = np.asarray(img)
    pred = get_prediction(np_img)
    Image.fromarray(pred.visualization).save(f"pred_{img_path.name}")
    print(pred)
