import string
import random
from flask import Flask, request, jsonify
import json 
import os
from PIL import Image
from utils.MainHR import CalcHR
from utils.tongueSeg import TongueProcess
from utils.quality import getQuality


app = Flask(__name__)



def randStr(chars = string.ascii_uppercase + string.digits, N=10):
	return ''.join(random.choice(chars) for _ in range(N))


@app.route('/tongue', methods=['POST'])
def tongue():
    if request.method == 'POST':
        try:
            seg = TongueProcess()
            img = Image.open(request.files['image'])
            tongue = seg.getTongueInfo(img)
            return jsonify(tongue)
        except Exception as e:
            return 400

@app.route("/quality", methods = ['POST'])
def get_quality():
    if request.method == 'POST':
        try:
            img = Image.open(request.files['image'])
            quality = getQuality(img)
            return jsonify({'quality':'good'}) if quality else (jsonify({'quality':'poor', 'error':'The image has bad quality'}),400)
        except Exception as e:
            return 400



@app.route("/heartrate", methods=['POST'])
def test():
    if request.method == 'POST':
        try:
            video = request.files['video']
            dir = './data/'+ str(video.name) + randStr()
            video.save(dir)
            bpm = CalcHR(dir).getHR()
            if os.path.exists(dir):
                os.remove(dir)
            return jsonify({"value":int(bpm)})
        except Exception as e:
            return 400


if __name__ == '__main__':
    app.run(app, port=5000)
