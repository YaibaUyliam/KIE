import requests
from Crypto.Cipher import AES
import hashlib
import numpy as np
import cv2
import time


def decrypt(source_key: str, data: bytes):
    source_key = source_key.split(",")

    k0 = hashlib.md5(source_key[0].encode()).hexdigest()
    k1 = hashlib.md5(source_key[1].encode()).hexdigest()
    k2 = source_key[1][:3]

    key = k0 + k1 + k2
    key = hashlib.md5(key.encode()).hexdigest().encode()

    cipher = AES.new(key, AES.MODE_ECB)

    return cipher.decrypt(data)


class DataDefine:
    def __init__(self, data: dict) -> None:
        # self.file_size: int | None = data.get("file_size")
        self.url: str = data.get("url")
        self.source_key: str = data.get("source_key")
        self.download_img
        self.bank_code: str = data.get("bank_code")
        self.bank_name: str = data.get("bank_name")
        self.check_camera: int = data.get("check_camera")
        self.order_amount: float = data.get("order_amount")
        self.order_no: str = data.get("order_no")
        self.site_name: str = data.get("site_name")
        self.user_name: str = data.get("user_name")
        self.device_source: str = data.get("device_source")
        self.device_no: str = data.get("device_no")

        self.order_create_time: str = data.get("order_create_time")
        self.order_confirm_time: str = data.get("order_confirm_time")

        self.ocr_res = data.get("ocr_res_origin_new")
        self.text_bill = data.get("textByLine_new")

        self.text_info = {}
        self.bb_info = {}
        self.key_value = []

    @property
    def download_img(self):
        if self.url:
            for _ in range(5):
                try:
                    response = requests.get(
                        self.url, stream=True, verify=True, timeout=5
                    )

                    response.raw.decode_content = True
                    # self.file_size = int(response.headers.get("Content-Length", 0))

                    if "_xxl_" in self.url and self.source_key:
                        content = decrypt(
                            source_key=self.source_key,
                            data=response.content,
                        )
                        self.bytes_img = content
                    else:
                        self.bytes_img = response.content

                    assert (
                        type(self.bytes_img) is bytes and len(self.bytes_img) > 0
                    ), "invalid input 'img' in DecodeImage"
                    self.img_nd = cv2.imdecode(
                        np.frombuffer(self.bytes_img, dtype="uint8"), 1
                    )

                    break

                except:
                    self.bytes_img = None
                    self.img_nd = None
                    time.sleep(5)

    @property
    def info_save_db(self):
        return {
            "url": self.url,
            "order_no": self.order_no,
            "user_name": self.user_name,
            "bank_code": self.bank_code,
            "device_source": self.device_source,
            "device_no": self.device_no,
            "order_create_time": self.order_create_time,
            "ben_name": self.text_info.get("beneficiary_name_value"),
            "ben_name_loc": self.bb_info.get("beneficiary_name_value"),
            "ben_number": self.text_info.get("beneficiary_number_value"),
            "ben_number_loc": self.bb_info.get("beneficiary_number_value"),
            "ben_bank": self.text_info.get("beneficiary_bank_value"),
            "ben_bank_loc": self.bb_info.get("beneficiary_bank_value"),
            "payer_name": self.text_info.get("payer_name_value"),
            "payer_name_loc": self.bb_info.get("payer_name_value"),
            "payer_number": self.text_info.get("payer_number_value"),
            "payer_number_loc": self.bb_info.get("payer_number_value"),
            "payer_bank": self.text_info.get("payer_bank_value"),
            "payer_bank_loc": self.bb_info.get("payer_bank_value"),
            "trans_money": self.text_info.get("transfer_money"),
            "trans_money_loc": self.bb_info.get("transfer_money"),
            "trans_money_text": self.text_info.get("transfer_money_text_value"),
            "trans_money_text_loc": self.bb_info.get("transfer_money_text_value"),
            "balance": self.text_info.get("account_balance"),
            "balance_loc": self.bb_info.get("account_balance"),
            "trans_time": self.text_info.get("transfer_time_value"),
            "trans_time_loc": self.bb_info.get("transfer_time_value"),
            "serial": self.text_info.get("serial_number_value"),
            "serial_loc": self.bb_info.get("serial_number_value"),
            "others_serial": self.text_info.get("others_serial_number_value"),
            "others_serial_loc": self.bb_info.get("others_serial_number_value"),
            "key_value": self.key_value,
        }

    @property
    def info_cls_ocr(self):
        return {
            "payee_name": self.text_info.get("beneficiary_name_value"),
            "payee_acc": self.text_info.get("beneficiary_number_value"),
            "payee_bank": self.text_info.get("beneficiary_bank_value"),
            "payer_name": self.text_info.get("payer_name_value"),
            "payer_acc": self.text_info.get("payer_number_value"),
            "payer_bank": self.text_info.get("payer_bank_value"),
            "trans_money": self.text_info.get("transfer_money"),
            "trans_money_text": self.text_info.get("transfer_money_text_value"),
            "balance": self.text_info.get("account_balance"),
            "trans_time": self.text_info.get("transfer_time_value"),
            "seri_number": self.text_info.get("serial_number_value"),
            "others_serial": self.text_info.get("others_serial_number_value"),
        }