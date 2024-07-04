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
        self.url: str | None = data.get("url")
        self.source_key: str | None = data.get("source_key")
        self.download_img
        self.bank_code: str | None = data.get("bank_code")
        self.bank_name: str | None = data.get("bank_name")
        self.check_camera: int | None = data.get("check_camera")
        self.order_amount: float | None = data.get("order_amount")
        self.order_no: str | None = data.get("order_no")
        self.site_name: str | None = data.get("site_name")
        self.user_name: str | None = data.get("user_name")
        self.device_source: str | None = data.get("device_source")
        self.device_no: str | None = data.get("device_no")

        self.order_create_time: str | None = data.get("order_create_time")

        self.info_text = {}

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
            "ben_name": self.info_text.get("beneficiary_name_value"),
            # "ben_name_points": self.beneficiary_name.get("points"),
            "ben_number": self.info_text.get("beneficiary_number_value"),
            # # "ben_number_points": self.beneficiary_number.get("points"),
            "ben_bank": self.info_text.get("beneficiary_bank_value"),
            # # "ben_bank_points": self.beneficiary_bank.get("points"),
            "payer_name": self.info_text.get("payer_name_value"),
            # # "payer_name_points": self.payer_name.get("points"),
            "payer_number": self.info_text.get("payer_number_value"),
            # # "payer_number_points": self.payer_number.get("points"),
            "payer_bank": self.info_text.get("payer_bank_value"),
            # # "payer_bank_points": self.payer_bank.get("points"),
            "trans_money": self.info_text.get("transfer_money"),
            # # "trans_money_points": self.transfer_money.get("points"),
            "trans_money_text": self.info_text.get("transfer_money_text_value"),
            "balance": self.info_text.get("account_balance"),
            # # "trans_money_text_points": self.transfer_money_text.get("points"),
            "trans_time": self.info_text.get("transfer_time_value"),
            # # "trans_time_points": self.transfer_time.get("points"),
            "serial": self.info_text.get("serial_number_value"),
            # # "serial_points": self.serial_number.get("points"),
            "others_serial": self.info_text.get("others_serial_number_value"),
            # # "others_serial_points": self.serial_number.get("points"),
            "phone_time": self.info_text.get("phone_time"),
            # # "phone_time_points": self.phone_time.get("points"),
        }
