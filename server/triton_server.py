import numpy as np
import argparse
from typing import Any, List
import logging
import json
import yaml
import time
from dotenv import load_dotenv
import os

from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton  # , TritonConfig
from pytriton.decorators import batch  # , fill_optionals

from ocr import OcrEngine
from PaddleOCR.tools.infer_kie_token_ser import SerPredictorV2
from PaddleOCR.tools.infer_kie_token_ser_re import SerRePredictor


config = load_dotenv()
test_env = os.environ["TEST_ENV"].lower() == "true"


def init_ser_model(cfg):
    return SerPredictorV2(cfg)


def init_re_model(cfg, cfg_ser):
    return SerRePredictor(cfg, cfg_ser)


def convert_to_python_float(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    return obj


class _InferFuncWrapper:
    def __init__(self, ocr_model, ser_model, re_model, device: str):
        self._ocr_model = ocr_model
        self._ser_model = ser_model
        self._re_model = re_model
        self._device = device

    # @fill_optionals(get_info=np.array([False], dtype=np.int8))
    @batch
    def __call__(self, image: np.ndarray) -> dict:
        data = {}
        data["image"] = image[0]
        data["ocr_info"] = self._ocr_model(data)

        ser_res, _ = self._ser_model(data.copy())
        re_res = self._re_model(data.copy())

        return {
            "ser_res": np.array([json.dumps(ser_res, default=convert_to_python_float)]),
            "re_res": np.array([json.dumps(re_res)])
        }


def _infer_function_factory(devices: List[str]):
    cfg_ser = yaml.safe_load(open("cfg/ser/ser_vi_layoutxlm.yaml"))
    cfg_cer_for_re = yaml.safe_load(open("cfg/re/ser_vi_layoutxlm_xfund_zh.yml"))
    cfg_re = yaml.safe_load(open("cfg/re/re_vi_layoutxlm_xfund_zh.yml"))

    infer_funcs = []
    for device in devices:
        ocr_model = OcrEngine(cfg_ser["Global"])
        ser_model = init_ser_model(cfg_ser)
        re_model = init_re_model(cfg_re, cfg_cer_for_re)

        infer_funcs.append(
            _InferFuncWrapper(ocr_model, ser_model, re_model, device=device)
        )

    return infer_funcs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=1,
        help="Batch size of request.",
        required=False,
    )
    parser.add_argument(
        "--number-of-instances",
        type=int,
        default=2,
        help="Batch size of request.",
        required=False,
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
    )

    devices = ["cuda:0"] * int(os.environ["NUMBER_OF_INSTANCES"])
    # devices = ["cuda:0", "cuda:0"]

    # with Triton(config=TritonConfig(log_verbose=3)) as triton:
    with Triton() as triton:
        triton.bind(
            model_name="KIE",
            infer_func=_infer_function_factory(devices),
            inputs=[
                Tensor(name="image", dtype=np.uint8, shape=(-1, -1, 3)),
            ],
            outputs=[
                Tensor(name="ser_res", dtype=bytes, shape=(-1,)),
                Tensor(name="re_res", dtype=bytes, shape=(-1,)),
            ],
            config=ModelConfig(
                max_batch_size=1,
            ),
            strict=False,
        )
        # logging.info("Serving model")
        triton.serve()


if __name__ == "__main__":
    main()
