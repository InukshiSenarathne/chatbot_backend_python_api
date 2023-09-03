import cv2 as cv2
from flask import current_app
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

class ImageModel:
    @staticmethod
    def get_init():
        # Implement your data retrieval logic here
        return {'message': 'image model resources init completed!'}

    def predict_image(image_file):
        MODEL_PATH = current_app.config['IMAGE_MODEL_PATH']
        VOICE_CALSSES = current_app.config['IMAGE_CALSSES']

        model = load_model(MODEL_PATH)
        img_size = (224, 224)
        fpath = image_file
        img = plt.imread(fpath)
        # resize the image so it is the same size as the images the model was trained on
        img = cv2.resize(img, img_size)  # in earlier code img_size=(224,224) was used for training the model
        img = np.expand_dims(img, axis=0)
        # now predict the image
        pred = model.predict(img)
        # this dataset has 15 classes so model.predict will return a list of 15 probability values
        # we want to find the index of the column that has the highest probability
        index = np.argmax(pred[0])
        # to get the actual Name of the class earlier Imade a list of the class names called classes
        klass = VOICE_CALSSES[index]

        # lets get the value of the highest probability
        probability = pred[0][index] * 100

        return klass, probability
