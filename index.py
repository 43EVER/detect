import io
import json
import base64
import requests
import sys

from PIL import Image
from ultralytics import YOLO
from flask import Flask, request, jsonify

app = Flask(__name__)

# base64 编码图像
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
    encoded_data = base64.b64encode(image_data)
    return encoded_data.decode("utf-8")

# base64 解码图像
def decode_image(encoded_data):
    image_data = base64.b64decode(encoded_data)
    image = Image.open(io.BytesIO(image_data))
    return image

# yolo白斑检测
def get_maskimage(image):
    model = YOLO("./best.pt")
    result = model(image, imgsz=1280, device='cpu')[0]
    image_array = result = result.plot(labels=False, boxes=True)
    image = Image.fromarray(image_array[..., ::-1])
    return image

@app.route('/process_json', methods=['GET', 'POST'])
def process_json():
    json_post = json.loads(request.get_data())
    image = decode_image(json_post['image_base64'])
    image_yolo = get_maskimage(image)
    image_yolo.save('result.jpg')
    image_base64 = encode_image('result.jpg')

    # 回传 json 包
    data = {
        'image_base64': image_base64,
        'area': '6.66', # 面积
        'region': 'A1', # 区域
    }
    json_response = json.dumps(data)
    return json_response, 200, {"Content-Type":"application/json"}

@app.route('/', methods=['GET'])
def index():
    # 构建测试 json 包
    image_base64 = encode_image('./test.jpg')
    data = {
        'image_base64': image_base64
    }
    json_post = json.dumps(data)
    response = requests.post(f'http://{sys.argv[1]}:{sys.argv[2]}/process_json', data=json_post)

    # 拿到回传结果解析图像
    json_result = json.loads(response.text)
    image = decode_image(json_result['image_base64'])
    image.save('result1.jpg')
    return 'hello word'