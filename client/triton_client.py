import numpy as np
import json
from yaml import safe_load
import os
import pickle

from pytriton.client import ModelClient

from preprocess import rm_hidden_letter, rm_stamp
from postprocess import SERPostProcessing, REPostProcessing
from utils.data_define import DataDefine
from utils.mongo import Mongo


def infer(img: np.ndarray, ocr_res: list, des="kie_server"):
    with ModelClient(des, "KIE", init_timeout_s=80) as client:
        ocr_res = pickle.dumps(ocr_res, protocol=pickle.HIGHEST_PROTOCOL)
        ocr_res = np.array([ocr_res])

        res_ocr = client.infer_sample(img, ocr_res)

    return res_ocr


class KieClient:
    def __init__(self, config_path: str) -> None:
        self.ser_postprocess = SERPostProcessing()
        self.re_postprocess = REPostProcessing()

        with open(config_path) as f:
            cfg = safe_load(f)["test"]["KIE_db"]

        self.mongo_db = Mongo(cfg=cfg, enable_signal=False)
        self.name_table = cfg["table"]

    def __call__(self, data: DataDefine, test_env: bool) -> None:
        # Preprocessing image
        if data.bank_code in ["101010", "113010"] and data.check_camera == 0:
            img = rm_hidden_letter(data.img_nd)
        elif data.bank_code in ["158010"] and data.check_camera == 0:
            img = rm_stamp(data.img_nd)
        else:
            img = data.img_nd

        res_server = infer(img, data.ocr_res, os.environ.get("IP_DEST"))
        ser_res = json.loads(res_server["ser_res"])
        re_res = json.loads(res_server["re_res"])

        data.text_info, data.bb_info = self.ser_postprocess(
            ser_res[0], data.bank_code, img, data.text_bill
        )
        data.key_value = self.re_postprocess(re_res, img)

        # if not test_env:
        #     self.mongo_db.insert_one(
        #         collection=self.name_table, document=data.info_save_db
        #     )

        return data.info_cls_ocr, data.info_save_db, data.key_value


if __name__ == "__main__":
    import cv2
    import requests
    import time
    from pymongo import MongoClient

    # response = requests.get(
    #     "https://ktpbds.aratalife.com/nfsxxf/cxwap01/fykzrm/2024/0701/585fe26649e146c5a37e84168516474e.jpeg",
    #     stream=True,
    #     verify=True,
    #     timeout=5,
    # )

    client = MongoClient(
        "mongodb://admin:92F767B6302F76A799A75447006AA59A@16.163.245.213:27017",
        serverSelectionTimeoutMS=5000,
    )
    db = client["ai-team"]
    collection = db["classify_ocr"]

    query = {
        "url": "https://ktpbds.aratalife.com/nfsxxf/cxwap01/v5ynpv/2024/0629/811c916489254adf95ed4a92cfc9e962.jpg"
    }
    sort_order = [("_id", -1)]
    projection = {}

    order_list_dest = list(
        collection.find(query, projection).sort(sort_order).limit(10)
    )

    client_kie = KieClient(config_path="cfg/mongo.yaml")

    for item in order_list_dest:
        info = DataDefine(item)

        client_kie(info, True)
        # img_nd = cv2.imdecode(np.frombuffer(bytes_img, dtype="uint8"), 1)

        # re_postprocess = REPostProcessing()
        # ser_postprocess = SERPostProcessing()

        # time_s = time.time()
        # res_server = send_request(img_nd, des="172.19.16.45")
        # ser_res = json.loads(res_server["ser_res"])
        # re_res = json.loads(res_server["re_res"])
        # print(time.time() - time_s)

        # key_value = re_postprocess(re_res, img_nd)
        # text_info, bb_info = ser_postprocess(ser_res[0], "243010", img_nd)
        # print(time.time() - time_s)

        # print(text_info)
        # print(bb_info)

        # with open("./cfg") as f:
        #     cfg = safe_load(f)["test"]["KIE_db"]

        # mongo_db = Mongo(cfg=cfg, enable_signal=False)
        # name_table = cfg["table"]

        # mongo_db.insert_one(
        #             collection=name_table, document=data.info_save_db
        #         )
