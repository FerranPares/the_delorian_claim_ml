import numpy as np
import torch
from mmengine.visualization import Visualizer


def draw_bboxes_onto_image(np_img: np.ndarray, bboxes: list[list[int, int, int, int]]) -> np.ndarray:
    """
    Get an input image and bboxes positions and returns the image with bboxes draw onto it.
    Inputs:
     - np.ndarray image with RGB channel order.
     - list of bboxes. Each bbox onsist in a list[int, int, int, int] formatted as [xyxy]
    """
    visualizer = Visualizer(image=np_img)
    visualizer.draw_bboxes(torch.tensor(bboxes))
    return visualizer.get_image()