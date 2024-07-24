import numpy as np

from PaddleOCR.paddleocr import PaddleOCR


def trans_poly_to_bbox(poly):
    x1 = int(np.min([p[0] for p in poly]))
    x2 = int(np.max([p[0] for p in poly]))
    y1 = int(np.min([p[1] for p in poly]))
    y2 = int(np.max([p[1] for p in poly]))
    return [x1, y1, x2, y2]


def convert(ocr_result):
    ocr_info = []

    if ocr_result is None:
        return ocr_info

    for res in ocr_result:
        ocr_info.append(
            {
                "transcription": res[1][0],
                "bbox": trans_poly_to_bbox(res[0]),
                "points": res[0],
            }
        )

    return ocr_info


def run_ocr(cfg, img):
    ocr_engine = PaddleOCR(
        use_angle_cls=True,
        show_log=False,
        rec_model_dir=cfg.get("kie_rec_model_dir"),
        det_model_dir=cfg.get("kie_det_model_dir"),
        cls_model_dir=cfg.get("kie_cls_model_dir"),
        rec_char_dict_path=cfg.get("kie_char_dict_path"),
        use_gpu=cfg["use_gpu"],
    )

    ocr_result = ocr_engine.ocr(img, cls=True)[0]

    ocr_info = []
    if ocr_result is None:
        return ocr_info

    for res in ocr_result:
        ocr_info.append(
            {
                "transcription": res[1][0],
                "bbox": trans_poly_to_bbox(res[0]),
                "points": res[0],
            }
        )

    return ocr_info


class OcrEngine:
    def __init__(self, cfg: dict) -> None:
        self.ocr_engine = PaddleOCR(
            use_angle_cls=True,
            show_log=False,
            rec_model_dir=cfg.get("kie_rec_model_dir"),
            det_model_dir=cfg.get("kie_det_model_dir"),
            cls_model_dir=cfg.get("kie_cls_model_dir"),
            rec_char_dict_path=cfg.get("kie_char_dict_path"),
            use_gpu=cfg["use_gpu"],
        )

    def __call__(self, data: dict) -> list[dict]:
        ocr_result = self.ocr_engine.ocr(data["image"], cls=True)[0]

        ocr_info = []
        if ocr_result is None:
            return ocr_info

        for res in ocr_result:
            ocr_info.append(
                {
                    "transcription": res[1][0],
                    "bbox": trans_poly_to_bbox(res[0]),
                    "points": res[0],
                }
            )

        return ocr_info
