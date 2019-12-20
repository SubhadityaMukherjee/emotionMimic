import os
import cv2
from base_camera import BaseCamera
import keras
from keras.models import load_model
import numpy as np
from PIL import Image
import os
import random


model = load_model("model/model.hdf5")
emotion_dict = {
    'angry': 0,
    'sad': 5,
    'neutral': 4,
    'disgust': 1,
    'surprise': 6,
    'fear': 2,
    'happy': 3
}

file_dict = {}
for a in emotion_dict:
    temp = os.listdir(f'emotions/{a}/')
    file_dict[a] =temp

label_map = dict((v, k) for k, v in emotion_dict.items())
face_cascade = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')

class Camera(BaseCamera):
    video_source = 0

    def __init__(self):
        if os.environ.get('OPENCV_CAMERA_SOURCE'):
            Camera.set_video_source(int(os.environ['OPENCV_CAMERA_SOURCE']))
        super(Camera, self).__init__()

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
        camera = cv2.VideoCapture(Camera.video_source)
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')
        temp_chck = 0
        while True:
            # read current frame
            _, img = camera.read()
            if temp_chck%5==0:

                face_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                try:
                    (x,y,w,h) = face_cascade.detectMultiScale(face_image, 1.3, 5)[0]


                    face_image = cv2.resize(face_image[y:y+h, x:x+w], (48, 48))


                    face_image = np.reshape(
                        face_image, [1, face_image.shape[0], face_image.shape[1], 1])
                    # cv2.imwrite('temp.jpg',face_image)

                    predicted_class = np.argmax(model.predict(face_image))
                    label = label_map[predicted_class]
                    # print(label)
                    # predimg = random.choice(file_dict[label])
                    # print(predimg)
                    # predimg = cv2.imread(f'emotions/{label}/{predimg}')

                    # predimg = cv2.resize(predimg, (48, 48))
                except Exception as e:
                    print(e)
                    predimg = img
                    label = ''
            temp_chck+=1
            # encode as a jpeg image and return it
            yield cv2.imencode('.jpg', img)[1].tobytes(), label
