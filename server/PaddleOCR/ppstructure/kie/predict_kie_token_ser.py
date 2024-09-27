# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../..")))

os.environ["FLAGS_allocator_strategy"] = "auto_growth"

import cv2

# import json
import numpy as np
import time

import tools.infer.utility as utility
from ppocr.data import create_operators, transform
from ppocr.postprocess import build_post_process
from ppocr.utils.logging import get_logger

# from ppocr.utils.visual import draw_ser_results
# from ppocr.utils.utility import get_image_file_list, check_and_read
from ppstructure.utility import parse_args

# from paddleocr import PaddleOCR

logger = get_logger()


class SerPredictor(object):
    def __init__(self, args, cfg):
        arch = cfg["Architecture"]
        args.kie_algorithm = arch["algorithm"]
        args.ser_model_dir = arch["Backbone"]["checkpoints"]
        args.ser_dict_path = cfg["PostProcess"]["class_path"]
        # args.use_onnx = True
        # args.use_gpu = False
        self.use_onnx = False
        # print(args.__dict__)

        pre_process_list = cfg["Eval"]["dataset"]["transforms"]

        self.preprocess_op = create_operators(pre_process_list, {"infer_mode": True})
        postprocess_params = cfg["PostProcess"]
        self.postprocess_op = build_post_process(postprocess_params)
        self.predictor, self.input_tensor, self.output_tensors, self.config = (
            utility.create_predictor(args, "ser", logger)
        )

        if self.use_onnx:
            self.input_names = [input.name for input in self.input_tensor]
            print(self.input_names)

        self.count = 0

    def __call__(self, data: dict):
        self.count += 1
        show_time_inference = self.count > 200
        if show_time_inference:
            time_s = time.time()

        batch = transform(data, self.preprocess_op)

        if batch[0] is None:
            return None, 0

        for idx in range(len(batch)):
            if isinstance(batch[idx], np.ndarray):
                batch[idx] = np.expand_dims(batch[idx], axis=0)
            else:
                batch[idx] = [batch[idx]]

        if not self.use_onnx:
            for idx in range(len(self.input_tensor)):
                self.input_tensor[idx].copy_from_cpu(batch[idx])

            self.predictor.run()

            outputs = []
            for output_tensor in self.output_tensors:
                output = output_tensor.copy_to_cpu()
                outputs.append(output)
                break

            preds = outputs[0]

        else:
            input_dict = {
                self.input_names[idx]: batch[idx]
                for idx in range(len(self.input_names))
            }
            preds = self.predictor.run([], input_dict)[0]

        post_result = self.postprocess_op(
            preds, segment_offset_ids=batch[6], ocr_infos=batch[7]
        )

        if show_time_inference:
            logger.info(f"Time inference SER: {time.time() - time_s}")
            self.count = 0

        self.predictor.try_shrink_memory()

        return post_result, batch


class SerPredictorV2(object):
    def __init__(self, args, cfg):
        arch = cfg["Architecture"]
        args.kie_algorithm = arch["algorithm"]
        args.ser_model_dir = "/home/yaiba/project/KIE/PaddleOCR/inference/ser_vi_layoutxlm_2308_onnx_1/model.onnx"
        args.ser_dict_path = cfg["PostProcess"]["class_path"]
        args.use_onnx = True
        # args.use_gpu = False
        self.use_onnx = True
        print(args.__dict__)

        pre_process_list = cfg["Eval"]["dataset"]["transforms"]

        self.preprocess_op = create_operators(pre_process_list, {"infer_mode": True})
        postprocess_params = cfg["PostProcess"]
        self.postprocess_op = build_post_process(postprocess_params)
        self.predictor, self.input_tensor, self.output_tensors, self.config = (
            utility.create_predictor(args, "ser", logger)
        )

        if self.use_onnx:
            self.input_names = [input.name for input in self.input_tensor]
            print(self.input_names)

    def __call__(self, data: dict):
        # time_s = time.time()
        batch = transform(data, self.preprocess_op)
        # logger.info(f"Time preprocessing: {time.time() - time_s}")

        if batch[0] is None:
            return None, 0

        for idx in range(len(batch)):
            if isinstance(batch[idx], np.ndarray):
                batch[idx] = np.expand_dims(batch[idx], axis=0)
            else:
                batch[idx] = [batch[idx]]

        if not self.use_onnx:
            for idx in range(len(self.input_tensor)):
                self.input_tensor[idx].copy_from_cpu(batch[idx])

            self.predictor.run()
            self.predictor.try_shrink_memory()

            outputs = []
            for output_tensor in self.output_tensors:
                output = output_tensor.copy_to_cpu()
                outputs.append(output)
                break

            preds = outputs[0]

        else:
            time_s = time.time()
            input_dict = {
                self.input_names[idx]: batch[idx]
                for idx in range(len(self.input_names))
            }
            preds = self.predictor.run([], input_dict)[0]
            logger.info(f"Time inference: {time.time() - time_s}")

        # time_s = time.time()
        post_result = self.postprocess_op(
            preds, segment_offset_ids=batch[6], ocr_infos=batch[7]
        )
        # logger.info(f"Time postprocessing: {time.time() - time_s}")

        return post_result, batch


def main():
    # image_file_list = get_image_file_list(args.image_dir)
    # ser_predictor = SerPredictor(args)
    # count = 0
    # total_time = 0

    # os.makedirs(args.output, exist_ok=True)
    # with open(
    #         os.path.join(args.output, 'infer.txt'), mode='w',
    #         encoding='utf-8') as f_w:
    #     for image_file in image_file_list:
    #         img, flag, _ = check_and_read(image_file)
    #         if not flag:
    #             img = cv2.imread(image_file)
    #             img = img[:, :, ::-1]
    #         if img is None:
    #             logger.info("error in loading image:{}".format(image_file))
    #             continue
    #         ser_res, _, elapse = ser_predictor(img)
    #         ser_res = ser_res[0]

    #         res_str = '{}\t{}\n'.format(
    #             image_file,
    #             json.dumps(
    #                 {
    #                     "ocr_info": ser_res,
    #                 }, ensure_ascii=False))
    #         f_w.write(res_str)

    #         img_res = draw_ser_results(
    #             image_file,
    #             ser_res,
    #             font_path=args.vis_font_path, )

    #         img_save_path = os.path.join(args.output,
    #                                      os.path.basename(image_file))
    #         cv2.imwrite(img_save_path, img_res)
    #         logger.info("save vis result to {}".format(img_save_path))
    #         if count > 0:
    #             total_time += elapse
    #         count += 1
    #         logger.info("Predict time of {}: {}".format(image_file, elapse))
    import yaml

    args = parse_args()
    cfg = yaml.safe_load(open("cfg/ser/ser_vi_layoutxlm.yaml"))
    model = SerPredictor(args, cfg)
    img = cv2.imread("/home/yaiba/Downloads/crcl2f-c4edebb8bb02cc-493105.jpg")
    post_result = model(img)
    logger.info(post_result)


if __name__ == "__main__":
    main()
