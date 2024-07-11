import json
import logging
import traceback
from dotenv import load_dotenv
from flask import Flask, request, jsonify
import requests
import os

import base64
import io
import cv2
from PIL import Image
import numpy as np

from utils.visual import draw_ser_results, draw_re_results
from pytriton.client import ModelClient

from postprocess import SERPostProcessing


load_dotenv()  # By default, load_dotenv doesn't override existing environment variables.

logging.basicConfig(
    level=logging.INFO,
    format=f"%(asctime)s %(levelname)s %(name)s : %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = "yaiba"


def infer(img: np.ndarray, des="kie_server"):
    with ModelClient(des, "KIE", init_timeout_s=80) as client:
        res_ocr = client.infer_sample(img)

    return res_ocr


def cv2_to_base64(img: np.ndarray):
    # _, im_arr = cv2.imencode(".jpg", img)

    # im_bytes = im_arr.tobytes()
    # im_b64 = base64.b64encode(img)
    _, buffer = cv2.imencode(".jpg", img)
    img_str = base64.b64encode(buffer).decode("utf-8")

    return img_str  # im_b64


def base64_to_cv2(base64_string):
    base64_string = base64_string.split(",")[1]

    imgdata = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(imgdata))

    return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)


ser_postprocess = SERPostProcessing()
base_head = "data:image/jpeg;base64,"


@app.route("/ser_re_visual", methods=["POST"])
def ser_re_visual():
    try:
        if request.mimetype == "multipart/form-data":
            data = request.files["file"]

            img_stream = io.BytesIO(data.read())
            img = Image.open(img_stream)
            img = np.array(img)[:, :, ::-1]

        elif request.mimetype == "application/json":
            data = request.json["url"]

            if "data:image" in data:
                img = base64_to_cv2(data)

            else:
                response = requests.get(data, timeout=(120, 120)).content
                img_bytes = io.BytesIO(response)
                img = Image.open(img_bytes)
                img = np.array(img)[:, :, :3][:, :, ::-1]

        else:
            return {"img_ser": None, "img_ser_post": None, "img_re": None}

        model_res = infer(img, os.environ.get("IP_DEST"))

        ser_res = json.loads(model_res["ser_res"])
        re_res = json.loads(model_res["re_res"])

        img_ser_res = None
        ser_res_post = None
        img_re_res = None
        if ser_res is not None:
            ser_res_post, _ = ser_postprocess(ser_res[0], None)
            img_draw_ser = draw_ser_results(
                img, ser_res[0], font_path="fonts/simfang.ttf"
            )
            img_ser_res = base_head + cv2_to_base64(img_draw_ser)

        if re_res is not None:
            img_draw_re = draw_re_results(img, re_res[0], font_path="fonts/simfang.ttf")
            img_re_res = base_head + cv2_to_base64(img_draw_re)

        return jsonify(
            {
                "img_ser": img_ser_res,
                "img_ser_post": ser_res_post,
                "img_re": img_re_res,
            }
        )

    except Exception as e:
        logger.error(e)
        logger.error(type(e).__name__)
        logger.error(traceback.format_exc())

        return {"img_ser": None, "img_ser_post": None, "img_re": None}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003, debug=False)
