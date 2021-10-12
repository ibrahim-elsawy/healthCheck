import string
import random
from flask_socketio import SocketIO, emit
from engineio.payload import Payload
from flask import Flask, request, jsonify
import json 
import os
from PIL import Image
from utils.MainHR import CalcHR
from utils.database import *
from utils.tongueSeg import TongueProcess
from utils.quality import getQuality

Payload.max_decode_packets = 500000000
# Payload.max_decode_packets = 5000000

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, max_http_buffer_size=1e12)
# socketio = SocketIO(app)



@socketio.on('connect')
def connect(json_data):
    connection_id = insertConnection(0)
    emit('connection id', {'id': connection_id})


def ack(): 
	print('message was received!')

def randStr(chars = string.ascii_uppercase + string.digits, N=10):
	return ''.join(random.choice(chars) for _ in range(N))

@socketio.on('video')
def handle_my_custom_event(data):
    req = json.loads(data)
    bpm = None
    data_buffer = []
    try: 
        conn_id, index_frame = getConnection(req['id'])
    except Exception as e:
        # TODO []-logging 
        emit('error', json.dumps({"status":400, "msg":"invalid id connection"}))


    # TODO []-logging
    filename = './data/file' + str(conn_id) + '.json'
    try: 
        data_buffer = loadJson(filename) if os.path.exists(filename) else []
        hr = CalcHR(req['fps'], data_buffer, index_frame)
        bpm, index_frame, data_buffer = hr.getHR(req['chunk']['data'], req["length"], req["width"])
    except Exception as e:
        emit('error', json.dumps({"status":400, "msg": str(e)}))
    if bpm is not None:
        emit('HR reading', json.dumps({"heartRate":bpm}), callback=ack)
    storeJson(filename, data_buffer)
    updateConnection(conn_id, index_frame)


@app.route('/tongue', methods=['POST'])
def tongue():
    if request.method == 'POST':
        seg = TongueProcess()
        img = Image.open(request.files['image'])
        tongue = seg.getTongueInfo(img)
        return jsonify(tongue)

@app.route("/quality", methods = ['POST'])
def get_quality():
	if request.method == 'POST':
		img = Image.open(request.files['image'])
	quality = getQuality(img)

	return jsonify({'quality':'good'}) if quality else (jsonify({'quality':'poor', 'error':'The image has bad quality'}),400)

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
    socketio.run(app, port=5000)
