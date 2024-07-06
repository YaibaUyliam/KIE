import numpy as np
import json
from yaml import safe_load
from dotenv import load_dotenv
import os

from pytriton.client import ModelClient

# from ultralytics import YOLO

from get_bill_info import InfoPostProcessing, REPostProcessing
from utils.data_define import DataDefine
from utils.mongo import Mongo

# from utils.alert_info import Producer, send_result, create_info_alert
from utils.preprocessing import rm_hidden_letter, rm_stamp


config = load_dotenv()
test_env = os.environ["TEST_ENV"].lower() == "true"


def send_request(img: np.ndarray, des="kie_server"):
    with ModelClient(des, "KIE", init_timeout_s=80) as client:
        res_ocr = client.infer_sample(img)

    return res_ocr


class KieClient:
    def __init__(self, config_path: str) -> None:
        self.post_res = InfoPostProcessing()
        self.re_postprocess = REPostProcessing()

        with open(config_path) as f:
            cfg = safe_load(f)["test"]["KIE_db"]

        self.mongo_db = Mongo(cfg=cfg, enable_signal=False)
        self.name_table = cfg["table"]

    def __call__(self, info: DataDefine) -> None:
        # Preprocessing image
        if info.bank_code in ["101010", "113010"] and info.check_camera == 0:
            img = rm_hidden_letter(info.img_nd)
        elif info.bank_code in ["158010"] and info.check_camera == 0:
            img = rm_stamp(info.img_nd)
        else:
            img = info.img_nd

        res_server = send_request(img)
        ser_res = json.loads(res_server["ser_res"])
        re_res = json.loads(res_server["re_res"])

        info.info_text = self.post_res.process(ser_res[0], info.bank_code)
        info.couples = self.re_postprocess(re_res)

        if not test_env:
            self.mongo_db.insert_one(
                collection=self.name_table, document=info.info_save_db
            )

        return


if __name__ == "__main__":
    import cv2
    import requests

    response = requests.get(
        "https://ktpbds.aratalife.com/nfsxxf/cxwap01/fykzrm/2024/0701/585fe26649e146c5a37e84168516474e.jpeg",
        stream=True,
        verify=True,
        timeout=5,
    )

    response.raw.decode_content = True
    bytes_img = response.content

    img_nd = cv2.imdecode(np.frombuffer(bytes_img, dtype="uint8"), 1)
    res_server = send_request(img_nd, des="172.19.16.45")
    ser_res = json.loads(res_server["ser_res"])
    re_res = json.loads(res_server["re_res"])
    

    re_postprocess = REPostProcessing()
    couples = re_postprocess(re_res, img_nd)