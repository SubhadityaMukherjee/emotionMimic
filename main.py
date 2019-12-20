from flask import Flask, render_template, Response
from time import time
from camera import Camera
import cv2
import os
import random
app = Flask(__name__)
label = ''

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
    # print(temp)
    file_dict[a] = temp


@app.route('/')
def index():
    return render_template('index.html')


def gen(camera):
    global label
    while True:
        frame = camera.get_frame()
        label = frame[1]

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame[0] + b'\r\n')


@app.route('/video_feed')
def video_feed():
    back = gen(Camera())
    print(back)
    return Response(back, mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/disp')
def disp():
    global label
    # back = gen(Camera())
    # return Response(back,mimetype='multipart/x-mixed-replace; boundary=frame')
    print(label)
    predimg = random.choice(file_dict[label])
    # print(predimg)
    predimg = cv2.imread(f'emotions/{label}/{predimg}')
    image = cv2.cvtColor(predimg, cv2.COLOR_BGR2RGB)
    np_img = Image.fromarray(image)
    img_encoded = image_to_byte_array(np_img)
    print('disp')


    return Response(predimg, status=200, mimetype="image/jpeg")


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
