import numpy as np
from mmdet.apis import DetInferencer
from dataclasses import dataclass

# Initialize the DetInferencer
inferencer = DetInferencer(model='config/DCN_plus_cfg.py', weights='weights/epoch_24.pth')
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
    label: int
    score: float
    bbox: list[int, int, int, int]
    
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
        
        return cls(
            label=pred['labels'][0],
            score=pred['scores'][0],
            bbox=pred['bboxes'][0]
        )

def get_prediction(np_img: np.ndarray) -> dict:
    # Perform inference
    mmdet_prediction_dict = inferencer(np_img)
    # Post-process and instantiates Prediction object
    return Prediction.from_mmdet_prediction(mmdet_prediction_dict)


