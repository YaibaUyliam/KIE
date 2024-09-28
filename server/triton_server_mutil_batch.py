import numpy as np
import copy
from typing import Any, List
import traceback
import json
import yaml
import time
from dotenv import load_dotenv
import os
import pickle
from loguru import logger
import subprocess

from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton, TritonConfig
from pytriton.decorators import batch  # , fill_optionals
from pytriton.exceptions import PyTritonUnrecoverableError
from pytriton.model_config import DynamicBatcher

from ocr import run_ocr, convert
from PaddleOCR.tools.infer_kie_token_ser import SerPredictorV2
from PaddleOCR.tools.infer_kie_token_ser_re import SerRePredictor
import paddle

load_dotenv()

cfg_ser = yaml.safe_load(open("cfg/ser/ser_vi_layoutxlm.yaml"))
cfg_cer_for_re = yaml.safe_load(open("cfg/re/ser_vi_layoutxlm_xfund_zh.yml"))
cfg_re = yaml.safe_load(open("cfg/re/re_vi_layoutxlm_xfund_zh.yml"))


def init_ser_model(cfg):
    return SerPredictorV2(cfg)


def init_re_model(cfg, cfg_ser):
    return SerRePredictor(cfg, cfg_ser)


def convert_to_python_float(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    return obj


class _InferFuncWrapper:
    def __init__(self, ser_model, re_model, device: str):
        # self._ocr_model = ocr_model
        self._ser_model = ser_model
        self._re_model = re_model
        self._device = device

    # @fill_optionals(get_info=np.array([False], dtype=np.int8))
    @batch
    def __call__(self, image: np.ndarray, ocr: np.object_) -> dict:
        batch_size = image.shape[0]

        ser_res = [None] * batch_size
        re_res = [None] * batch_size
        ser_res_other = [None] * batch_size

        if len(ocr) != 0:
            try:
                batch = []

                for idx in range(image.shape[0]):
                    data = {}
                    data["image"] = image[idx]

                    ocr_format = pickle.loads(ocr[idx][0])
                    data["ocr_info"] = convert(ocr_format[0])

                    batch.append(data)

                print("batch size: ", len(batch))

                ser_res, _ = self._ser_model(copy.deepcopy(batch))
                # re_res, ser_res_other = self._re_model(copy.deepcopy(batch))

            except Exception as e:
                logger.error(traceback.format_exc())
                # logger.exception(e)

                if type(e).__name__ == "OSError":
                    output = subprocess.check_output(["nvidia-smi"])
                    output = output.decode("utf-8")
                    logger.info(output)

                    raise PyTritonUnrecoverableError(
                        "Some unrecoverable error occurred, "
                        "thus no further inferences are possible."
                    ) from e

        # time.sleep(0.07)
        # paddle.device.cuda.empty_cache()
        return {
            "ser_res": np.array([json.dumps(res, default=convert_to_python_float) for res in ser_res]), # fmt: skip
            "re_res": np.array([json.dumps(res, default=convert_to_python_float) for res in re_res]), # fmt: skip
            "ser_res_other": np.array([json.dumps(res, default=convert_to_python_float) for res in ser_res_other]), # fmt: skip
        }


def _infer_function_factory(devices: List[str]):
    infer_funcs = []

    for device in devices:
        # ocr_model = OcrEngine(cfg_ser["Global"])
        ser_model = init_ser_model(cfg_ser)
        re_model = init_re_model(cfg_re, cfg_cer_for_re)

        infer_funcs.append(_InferFuncWrapper(ser_model, re_model, device=device))

    return infer_funcs


def main():
    devices = ["cuda:0"] * int(os.environ["NUMBER_OF_INSTANCES"])
    # devices = ["cuda:0", "cuda:0"]

    # with Triton(config=TritonConfig(log_verbose=3)) as triton:
    with Triton() as triton:
        triton.bind(
            model_name="KIE",
            infer_func=_infer_function_factory(devices),
            inputs=[
                Tensor(name="image", dtype=np.uint8, shape=(-1, -1, 3)),
                Tensor(name="ocr", dtype=bytes, shape=(-1,)),
            ],
            outputs=[
                Tensor(name="ser_res", dtype=bytes, shape=(-1,)),
                Tensor(name="re_res", dtype=bytes, shape=(-1,)),
                Tensor(name="ser_res_other", dtype=bytes, shape=(-1,)),
            ],
            config=ModelConfig(
                max_batch_size=4,
                batcher=DynamicBatcher(max_queue_delay_microseconds=500000, preferred_batch_size=[2, 4])
            ),
            strict=False,
        )
        triton.serve()


if __name__ == "__main__":
    main()
    # ser_model = init_ser_model(cfg_ser)
    # re_model = init_re_model(cfg_re, cfg_cer_for_re)

    # import json
    # import requests
    # import cv2

    # batch_test = []
    # with open("/home/yaiba/project/kie/client/test.json") as f:
    #     data_test = json.load(f)

    # for idx in range(2):
    #     data = {}

    #     url = data_test[idx]["origin_url"]
    #     print(url)
    #     response = requests.get(url, stream=True, verify=True, timeout=5)

    #     response.raw.decode_content = True
    #     bytes_img = response.content

    #     img_nd = cv2.imdecode(np.frombuffer(bytes_img, dtype="uint8"), 1)

    #     data["image"] = img_nd

    #     ocr_format = data_test[idx][
    #         "ocr_origin_strange_font"
    #     ]  # pickle.loads(ocr[0][idx])
    #     data["ocr_info"] = convert(ocr_format[0])

    #     batch_test.append(data)

    # ser_res, _ = re_model(batch_test.copy())
    # for v in ser_res:
    #     print(ser_res)
    #     print("------------")
