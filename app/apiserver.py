import base64
import PIL
import re
from io import BytesIO

from flask import Flask, jsonify, request
from flask_cors import CORS
from app import get_points_with_draw, segment_with_points

app = Flask(__name__)
CORS(app)


@app.route('/food_segmentation', methods=['POST'])
def food_segmentation():
    imageB64 = re.sub('^data:image/.+;base64,', '', request.json['imageB64'])
    x1 = request.json['x1']
    y1 = request.json['y1']
    x2 = request.json['x2']
    y2 = request.json['y2']

    if x1 == None or y1 == None or x2 == None or y2 == None:
        result = {
            "imageB64": request.json['imageB64']
        }
        return jsonify(result)
    image = PIL.Image.open(BytesIO(base64.b64decode(imageB64)))

    # print(image)

    get_points_with_draw(None, "Add Mask", x1+50, y1+50)
    get_points_with_draw(None, "Add Mask", x2-50, y1+50)
    get_points_with_draw(None, "Add Mask", x1+50, y2-50)
    get_points_with_draw(None, "Add Mask", x2-50, y2-50)

    fig, _ = segment_with_points(image)

    img_byte_array = BytesIO()
    fig.save(img_byte_array, format="PNG")
    img_byte_array = img_byte_array.getvalue()

    # Bytes를 base64로 인코딩
    base64_encoded = "data:image/png;base64," + base64.b64encode(img_byte_array).decode("utf-8")

    result = {
        "imageB64": base64_encoded
    }
    return jsonify(result)


if __name__ == '__main__':
    ip_address = "127.0.0.1"
    port_number = 8001
    app.run(ip_address, port=int(port_number), debug=True)
