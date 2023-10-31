import json
import numpy as np
from pathlib import Path
from keras.applications.vgg16 import VGG16
from keras.applications.imagenet_utils import preprocess_input



vgg16 = VGG16(weights='imagenet')

valid_categories = ["car_wheel", "ambulance", "beach_wagon", "cab", "convertible", "jeep", "limousine", "minivan", "model_t", "racer", "sports_car"]

CLASS_INDEX_PATH = str(Path(__file__).parent / 'imagenet_class_index.json')


def prepare_image_224(np_img: np.ndarray) -> np.ndarray:
    x = np.expand_dims(np_img, axis=0)
    x = preprocess_input(x)
    return x


def get_predictions(preds, top=5):
    
    with open(CLASS_INDEX_PATH, 'r') as fp:
        CLASS_INDEX = json.load(fp)
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
        result.sort(key=lambda x: x[2], reverse=True)
        results.append(result)
    return results

def pipe1(np_img: np.ndarray) -> bool:
    img = prepare_image_224(np_img)
    out = vgg16.predict(img)
    preds = get_predictions(out, top=1)
    prediction_class_name = preds[0][0][1]
    return prediction_class_name in valid_categories
