import json
import logging
import traceback
from dotenv import load_dotenv
from flask import Flask, request, jsonify
import requests
import os
import pickle
import time

import base64
import io
import cv2
from PIL import Image
import numpy as np

from pytriton.client import ModelClient
from pymongo import MongoClient

from postprocess import SERPostProcessing, SEROtherPostProcessing
from utils.visual import draw_ser_results, draw_re_results


load_dotenv()  # By default, load_dotenv doesn't override existing environment variables.

logging.basicConfig(
    level=logging.INFO,
    format=f"%(asctime)s %(levelname)s %(name)s : %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = "yaiba"


client = MongoClient(
    os.environ.get("URL_DB"),
    serverSelectionTimeoutMS=5000,
)
db = client["ai-team"]
collection = db["classify_ocr"]
projection = {
    "ocr_origin_strange_font": 1,
    "text_by_line_strange_font": 1,
}

limit = 1000
sort_order = [("_id", -1)]


def infer(img: np.ndarray, ocr_res: list, des="kie_server"):
    with ModelClient(des, "KIE", init_timeout_s=80) as client:
        ocr_res = pickle.dumps(ocr_res, protocol=pickle.HIGHEST_PROTOCOL)
        ocr_res = np.array([ocr_res])

        res_ocr = client.infer_sample(img, ocr_res)

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
ser_other_postprocess = SEROtherPostProcessing()
base_head = "data:image/jpeg;base64,"


@app.route("/ser_re_visual", methods=["POST"])
def ser_re_visual():
    try:
        if request.mimetype == "multipart/form-data":
            data = request.files["file"]
            ocr = []
            text_only = None

            img_stream = io.BytesIO(data.read())
            img = Image.open(img_stream)
            img = np.array(img)[:, :, ::-1]

        elif request.mimetype == "application/json":
            data = request.json
            url = data["url"]
            # will query db in future
            logger.info(url)

            if "data:image" in url:
                img = base64_to_cv2(url)

            else:
                response = requests.get(url, timeout=(120, 120)).content
                img_bytes = io.BytesIO(response)
                img = Image.open(img_bytes)
                img = np.array(img)[:, :, :3][:, :, ::-1]

                time_s = time.time()
                if (
                    data.get("ocr_origin_strange_font") is not None
                    and data.get("text_by_line_strange_font") is not None
                ):
                    order_list = data

                else:
                    query = {"url": url}
                    order_list = list(collection.find(query, projection))[0]
                    logger.info("Time query DB: %s", time.time() - time_s)

                ocr = order_list.get("ocr_origin_strange_font", [])
                text_only = order_list.get("text_by_line_strange_font")

        else:
            return {
                "img_ser": None,
                "img_ser_post": None,
                "img_re": None,
                "img_ser_other": None,
            }

        time_s = time.time()
        model_res = infer(img, ocr, os.environ.get("IP_DEST"))
        logger.info("Time model infer: %s", time.time() - time_s)

        ser_res = json.loads(model_res["ser_res"])
        re_res = json.loads(model_res["re_res"])
        ser_res_other = json.loads(model_res["ser_res_other"])

        ser_res_other = ser_other_postprocess(ser_res_other, img)

        img_ser_res = None
        ser_res_post = None
        img_re_res = None
        if ser_res is not None:
            ser_res_post, _ = ser_postprocess(ser_res[0], None, img, text_only)
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
                "img_ser_other": ser_res_other,
            }
        )

    except Exception as e:
        logger.error(e)
        logger.error(type(e).__name__)
        logger.error(traceback.format_exc())

        return {
            "img_ser": None,
            "img_ser_post": None,
            "img_re": None,
            "img_ser_other": None,
        }


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003, debug=False)
