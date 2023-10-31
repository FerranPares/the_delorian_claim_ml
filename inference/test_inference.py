import numpy as np
import urllib.request
from keras.preprocessing.image import img_to_array, load_img
from inference.predict_keras_phase1 import pipe1



def url_to_npimg(img_url: str) -> np.ndarray:
    urllib.request.urlretrieve(img_url, 'image.jpg')
    img = load_img('image.jpg', target_size=(224,224))
    x = img_to_array(img)
    return x
    

if __name__ == "__main__":
    np_img = url_to_npimg("https://tse4.mm.bing.net/th?id=OIP.FaZela57De0uzfVxVY3JJQHaEo&pid=Api&P=0&w=289&h=181")
    prediction_result = pipe1(np_img)
    print(prediction_result)