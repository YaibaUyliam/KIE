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
from utils.visual import draw_ser_results, draw_re_results



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
    
        # connection_string = "mongodb://admin:92F767B6302F76A799A75447006AA59A@16.163.245.213:27017/ai-team"
        # client = MongoClient(connection_string)
        # db = client['ai-team']
        # self.collection = db[f'kie']

    def __call__(self, data: DataDefine, test_env: bool) -> None:
        img = data.img_nd

        res_server = infer(img, data.ocr_res, '0.0.0.0')
        ser_res = json.loads(res_server["ser_res"])
        re_res = json.loads(res_server["re_res"])

        # Draw
        img_draw_ser, img_draw_re = None, None
        if ser_res is not None:
            img_draw_ser = draw_ser_results(img, ser_res[0], font_path="fonts/simfang.ttf")
        if re_res is not None:
            img_draw_re = draw_re_results(img, re_res[0], font_path="fonts/simfang.ttf")

        # postprocess
        data.text_info, data.bb_info = self.ser_postprocess(
            ser_res[0], data.bank_code, img, data.text_bill
        )
        data.kie_re = self.re_postprocess(re_res, img)

        # self.collection.insert_one(
        #     {
        #         'kie_ser': data.info_save_db,
        #         'kie_re': data.key_value
        #     }
        # )
        # print(data.kie_ser)
        # print(data.kie_re)
        

        return data.order_no, data.kie_ser, data.kie_re, img_draw_ser, img_draw_re

def call_api(filename):
    try:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            order_no = filename.split('_')[0] if '-' in filename else filename.split('.')[0]

            if os.path.exists(f'{ouput_path}/{order_no}.jpg'):
                return

            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)

            data = {
                'img_nd': img,
                'bank_code': bank_code,
                'order_no': order_no,
            }
            info = DataDefine(data)
            order_no, kie_ser, kie_re, img_draw_ser, img_draw_re = client_kie(info, True)

            # Save img
            image1 = img_draw_ser if img_draw_ser is not None else img
            image2 = img_draw_re if img_draw_re is not None else img
            merged_image = cv2.hconcat([image2, image1])
            cv2.imwrite(f'{ouput_path}/{order_no}.jpg', merged_image)


            # Save txt 
            data_dict = {
                order_no: {
                    'kie_ser': kie_ser,
                    'kie_re': kie_re
                }
            }
            with open(txt_path, 'a', encoding='utf-8') as file:
                file.write(json.dumps(data_dict) + '\n')
    except:
        import traceback
        print(traceback.format_exc())


if __name__ == "__main__":
    # import cv2
    # import requests
    # import time
    # from pymongo import MongoClient

    # # Load txt file
    # file_path = '../zz[IN]_order_no.txt'
    # with open(file_path, 'r') as file:
    #     lines = [line.strip() for line in file.readlines()]
    # order_no_lines = lines
    # # print(order_no_lines)

    # # Query
    # client = MongoClient(
    #     "mongodb://admin:92F767B6302F76A799A75447006AA59A@16.163.245.213:27017",
    #     serverSelectionTimeoutMS=5000,
    # )
    # db = client["ai-team"]
    # collection = db["classify_ocr"]

    # query = {
    #     "order_no": {"$in": order_no_lines},
    # }
    # sort_order = [("_id", -1)]
    # projection = {}

    # order_list_dest = list(
    #     collection.find(query, projection).sort(sort_order).limit(10)
    # )

    # client_kie = KieClient(config_path="cfg/mongo.yaml")

    # for item in order_list_dest:
    #     info = DataDefine(item)

    #     client_kie(info, True)



    import cv2
    import os
    import json

    bank_code = '243020'
    folder_path = '/home/shaun/Music/hiro/image_zfbv2/image/real_1'
    txt_path = f'{folder_path}.txt'

    ouput_path = f'{folder_path}_[ser_re]'
    os.makedirs(ouput_path, exist_ok=True)

    client_kie = KieClient(config_path="cfg/mongo.yaml")  

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=20) as ex:
        [ex.submit(call_api, filename) for filename in os.listdir(folder_path)]