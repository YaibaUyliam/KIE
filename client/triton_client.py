import numpy as np
import json
from yaml import safe_load
import os
import pickle

from pytriton.client import ModelClient
from pymongo import MongoClient

from preprocess import rm_hidden_letter, rm_stamp
from postprocess import SERPostProcessing, REPostProcessing, SEROtherPostProcessing
from utils.data_define import DataDefine
from utils.mongo import Mongo
from utils.visual import draw_ser_results, draw_re_results


def infer(img: np.ndarray, ocr_res: list, des="kie_server"):
    with ModelClient(des, "KIE", init_timeout_s=80) as client:
        ocr_res = pickle.dumps(ocr_res, protocol=pickle.HIGHEST_PROTOCOL)
        ocr_res = np.array([ocr_res])

        res_ocr = client.infer_sample(img, ocr_res)

    return res_ocr


class KieClient:
    def __init__(self, connection_string: str = None) -> None:
        self.ser_postprocess = SERPostProcessing()
        self.re_postprocess = REPostProcessing()
        self.ser_other_postprocess = SEROtherPostProcessing()

        self.connection_string = connection_string
        if self.connection_string:
            client = MongoClient(connection_string)
            mydb = client["kie"]
            self.mycol = mydb["money_add"]

    def __call__(self, data: DataDefine) -> None:
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
        ser_res_other = json.loads(res_server["ser_res_other"])

        data.text_info, data.bb_info = self.ser_postprocess(
            ser_res[0], data.bank_code, img, data.text_bill
        )
        if re_res:
            data.key_value = self.re_postprocess(re_res, img)
            data.ser_other = self.ser_other_postprocess(ser_res_other, img)

        if self.connection_string:
            self.mycol.insert_one(document=data.info_save_db)

        return data.info_cls_ocr, data.info_save_db, data.key_value, data.ser_other


def call_api(filename):
    try:
        if filename.lower().endswith(
            (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif")
        ):
            order_no = (
                filename.split("_")[0] if "_" in filename else filename.split(".")[0]
            )

            if os.path.exists(f"{ouput_path}/{order_no}.jpg"):
                return

            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)

            data = {
                "img_nd": img,
                "bank_code": bank_code,
                "order_no": order_no,
            }
            info = DataDefine(data, mode="folder")
            order_no, kie_ser, kie_re, img_draw_ser, img_draw_re = client_kie(
                info, True
            )

            # Save img
            image1 = img_draw_ser if img_draw_ser is not None else img
            image2 = img_draw_re if img_draw_re is not None else img
            merged_image = cv2.hconcat([image2, image1])
            cv2.imwrite(f"{ouput_path}/{order_no}.jpg", merged_image)

            # Save txt
            output_dict = {
                "order_no": order_no,
                "bank_code": bank_code,
                "kie_ser": kie_ser,
                "kie_re": kie_re,
            }
            base_name = filename.split(".")[0]
            json_file_path = f"{json_output_path}/{base_name}.json"
            with open(json_file_path, "w", encoding="utf-8") as file:
                json.dump(output_dict, file, indent=4, ensure_ascii=False)
    except:
        import traceback

        print(traceback.format_exc())


if __name__ == "__main__":
    # Way 1: Run from DB
    import cv2
    import requests
    import time
    from pymongo import MongoClient

    # Load txt file
    file_path = "../zz[IN]_order_no.txt"
    with open(file_path, "r") as file:
        lines = [line.strip() for line in file.readlines()]
    order_no_lines = lines
    # print(order_no_lines)

    # Query
    client = MongoClient(
        "mongodb://admin:92F767B6302F76A799A75447006AA59A@16.163.245.213:27017",
        serverSelectionTimeoutMS=5000,
    )
    db = client["ai-team"]
    collection = db["classify_ocr"]

    query = {
        "order_no": {"$in": order_no_lines},
    }
    sort_order = [("_id", -1)]
    projection = {}

    order_list_dest = list(
        collection.find(query, projection).sort(sort_order).limit(10)
    )

    client_kie = KieClient(config_path="cfg/mongo.yaml")

    for item in order_list_dest:
        info = DataDefine(item, mode="db")

        client_kie(info, True)

    # # Way 2: Run from folder
    # import cv2
    # import os
    # import json

    # bank_code = '243020'
    # folder_path = '/home/shaun/Music/hiro/test/243020_1'

    # ouput_path = f'{folder_path}_[ser_re]'
    # os.makedirs(ouput_path, exist_ok=True)
    # json_output_path = f'{folder_path}_[json]'
    # os.makedirs(json_output_path, exist_ok=True)

    # client_kie = KieClient(config_path="cfg/mongo.yaml")

    # from concurrent.futures import ThreadPoolExecutor

    # with ThreadPoolExecutor(max_workers=20) as ex:
    #     [ex.submit(call_api, filename) for filename in os.listdir(folder_path)]
