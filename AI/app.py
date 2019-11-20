from flask import Flask,render_template
from flask import request
from detection import TextDetection
from recognition import TextRecognition
import cv2
import json
import os
from config.config import DETECTION_CONFIG, RECOGNITION_CONFIG
app = Flask(__name__)
@app.route('/ocr',methods=["POST"])
def hello_world():
    print("in here")
    if request.method == "POST":
        image_path = request.data.decode("utf-8")
        image = cv2.imread(image_path)
        result= recognition.recognize_text(image)
        return result
    return "ok"
if __name__ == '__main__':
    detection = TextDetection(DETECTION_CONFIG.MODEL_PATH, DETECTION_CONFIG.TEXT_THRESHOLD, DETECTION_CONFIG.LOW_TEXT, \
                              DETECTION_CONFIG.LINK_THRESHOLD, DETECTION_CONFIG.CUDA, DETECTION_CONFIG.CANVAS_SIZE, \
                              DETECTION_CONFIG.MAG_RATIO, DETECTION_CONFIG.POLY, DETECTION_CONFIG.SHOW_TIME)

    recognition = TextRecognition(detection,RECOGNITION_CONFIG)
    app.run(port=3000)