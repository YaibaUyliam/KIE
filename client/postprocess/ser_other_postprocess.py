import yaml
import numpy as np

from .util import fit_bbox_2


class SEROtherPostProcessing:
    def __init__(self) -> None:
        pass
    
    def check_same_bbox(self, bbox1, bbox2):
        return bbox1[0] == bbox2[0] and bbox1[1] == bbox2[1] and bbox1[2] == bbox2[2] and bbox1[3] == bbox2[3]

    def __call__(self, ser_res_other, image):
        '''
        INPUT:
            ser_res_other = [[{'transcription': '82.387%', 'bbox': [354, 17, 544, 43], 'points': [[354.0, 17.0], [544.0, 19.0], [544.0, 43.0], [354.0, 41.0]], 'pred_id': 0, 'pred': 'O'}, {'transcription': '17:52支8', 'bbox': [32, 21, 179, 40], 'points': [[32.0, 21.0], [179.0, 21.0], [179.0, 40.0], [32.0, 40.0]], 'pred_id': 0, 'pred': 'O'}, {'transcription': '單情', 'bbox': [59, 88, 175, 116], 'points': [[59.0, 88.0], [175.0, 88.0], [175.0, 116.0], [59.0, 116.0]], 'pred_id': 23, 'pred': 'HEADER'}, {'transcription': '陈文辰', 'bbox': [253, 261, 324, 289], 'points': [[253.0, 261.0], [324.0, 261.0], [324.0, 289.0], [253.0, 289.0]], 'pred_id': 21, 'pred': 'ANSWER'}, {'transcription': '-200.20', 'bbox': [202, 311, 369, 352], 'points': [[202.0, 311.0], [369.0, 311.0], [369.0, 352.0], [202.0, 352.0]], 'pred_id': 9, 'pred': 'MONEY'}, {'transcription': '交易成功', 'bbox': [244, 371, 332, 395], 'points': [[244.0, 371.0], [332.0, 371.0], [332.0, 395.0], [244.0, 395.0]], 'pred_id': 15, 'pred': 'CONFIRM'}, {'transcription': '服務費', 'bbox': [35, 437, 98, 463], 'points': [[35.0, 437.0], [98.0, 437.0], [98.0, 463.0], [35.0, 463.0]], 'pred_id': 19, 'pred': 'QUESTION'}, {'transcription': '0.20', 'bbox': [177, 439, 225, 463], 'points': [[177.0, 439.0], [225.0, 439.0], [225.0, 463.0], [177.0, 463.0]], 'pred_id': 21, 'pred': 'ANSWER'}, {'transcription': '創建時間', 'bbox': [35, 488, 118, 511], 'points': [[35.0, 488.0], [118.0, 488.0], [118.0, 511.0], [35.0, 511.0]], 'pred_id': 19, 'pred': 'QUESTION'}, {'transcription': '2024-08-0217:51:46', 'bbox': [181, 491, 374, 509], 'points': [[181.0, 491.0], [374.0, 491.0], [374.0, 509.0], [181.0, 509.0]], 'pred_id': 21, 'pred': 'ANSWER'}, {'transcription': '付款方式', 'bbox': [36, 536, 118, 560], 'points': [[36.0, 536.0], [118.0, 536.0], [118.0, 560.0], [36.0, 560.0]], 'pred_id': 19, 'pred': 'QUESTION'}, {'transcription': '中國工商銀行蓄卡（8542）>', 'bbox': [181, 537, 438, 560], 'points': [[181.0, 537.0], [438.0, 537.0], [438.0, 560.0], [181.0, 560.0]], 'pred_id': 21, 'pred': 'ANSWER'}, {'transcription': '轉到', 'bbox': [35, 585, 98, 609], 'points': [[35.0, 585.0], [98.0, 585.0], [98.0, 609.0], [35.0, 609.0]], 'pred_id': 19, 'pred': 'QUESTION'}, {'transcription': '湖南银行（1335）陈文辰', 'bbox': [177, 584, 404, 611], 'points': [[177.0, 584.0], [404.0, 584.0], [404.0, 611.0], [177.0, 611.0]], 'pred_id': 21, 'pred': 'ANSWER'}, {'transcription': '處理進度', 'bbox': [36, 635, 118, 659], 'points': [[36.0, 635.0], [118.0, 635.0], [118.0, 659.0], [36.0, 659.0]], 'pred_id': 19, 'pred': 'QUESTION'}, {'transcription': '提交銀行處理', 'bbox': [215, 635, 339, 659], 'points': [[215.0, 635.0], [339.0, 635.0], [339.0, 659.0], [215.0, 659.0]], 'pred_id': 21, 'pred': 'ANSWER'}, {'transcription': '08-0217:52', 'bbox': [217, 671, 321, 689], 'points': [[217.0, 671.0], [321.0, 671.0], [321.0, 689.0], [217.0, 689.0]], 'pred_id': 21, 'pred': 'ANSWER'}, {'transcription': '銀行處理中', 'bbox': [216, 717, 319, 740], 'points': [[216.0, 717.0], [319.0, 717.0], [319.0, 740.0], [216.0, 740.0]], 'pred_id': 21, 'pred': 'ANSWER'}, {'transcription': '08-0217:52', 'bbox': [216, 751, 323, 773], 'points': [[216.0, 751.0], [323.0, 751.0], [323.0, 773.0], [216.0, 773.0]], 'pred_id': 21, 'pred': 'ANSWER'}, {'transcription': '銀行入成功', 'bbox': [216, 799, 337, 821], 'points': [[216.0, 799.0], [337.0, 799.0], [337.0, 821.0], [216.0, 821.0]], 'pred_id': 21, 'pred': 'ANSWER'}, {'transcription': '08-0217:52', 'bbox': [217, 835, 320, 853], 'points': [[217.0, 835.0], [320.0, 835.0], [320.0, 853.0], [217.0, 853.0]], 'pred_id': 21, 'pred': 'ANSWER'}, {'transcription': '進度说明', 'bbox': [33, 878, 119, 906], 'points': [[34.0, 878.0], [119.0, 882.0], [118.0, 906.0], [33.0, 902.0]], 'pred_id': 19, 'pred': 'QUESTION'}, {'transcription': '银行反馈已入账，如有疑问请联系银行', 'bbox': [181, 883, 519, 905], 'points': [[181.0, 883.0], [519.0, 883.0], [519.0, 905.0], [181.0, 905.0]], 'pred_id': 21, 'pred': 'ANSWER'}, {'transcription': '客服', 'bbox': [181, 913, 224, 936], 'points': [[181.0, 913.0], [224.0, 913.0], [224.0, 936.0], [181.0, 936.0]], 'pred_id': 21, 'pred': 'ANSWER'}, {'transcription': '轉明', 'bbox': [35, 961, 118, 984], 'points': [[35.0, 961.0], [118.0, 961.0], [118.0, 984.0], [35.0, 984.0]], 'pred_id': 19, 'pred': 'QUESTION'}, {'transcription': '轉', 'bbox': [179, 961, 224, 985], 'points': [[179.0, 961.0], [224.0, 961.0], [224.0, 985.0], [179.0, 985.0]], 'pred_id': 21, 'pred': 'ANSWER'}, {'transcription': '支付', 'bbox': [35, 1011, 118, 1033], 'points': [[35.0, 1011.0], [118.0, 1011.0], [118.0, 1033.0], [35.0, 1033.0]], 'pred_id': 19, 'pred': 'QUESTION'}, {'transcription': '支付获5积分（已翻5倍）', 'bbox': [211, 1016, 410, 1039], 'points': [[211.0, 1016.0], [410.0, 1016.0], [410.0, 1039.0], [211.0, 1039.0]], 'pred_id': 21, 'pred': 'ANSWER'}, {'transcription': '訂單號', 'bbox': [35, 1071, 99, 1095], 'points': [[35.0, 1071.0], [99.0, 1071.0], [99.0, 1095.0], [35.0, 1095.0]], 'pred_id': 19, 'pred': 'QUESTION'}, {'transcription': '2024080220004001110028001707089', 'bbox': [180, 1073, 539, 1092], 'points': [[180.0, 1073.0], [539.0, 1073.0], [539.0, 1092.0], [180.0, 1092.0]], 'pred_id': 21, 'pred': 'ANSWER'}, {'transcription': '8', 'bbox': [180, 1104, 197, 1124], 'points': [[180.0, 1104.0], [197.0, 1104.0], [197.0, 1124.0], [180.0, 1124.0]], 'pred_id': 21, 'pred': 'ANSWER'}, {'transcription': '單管理', 'bbox': [35, 1200, 123, 1224], 'points': [[35.0, 1200.0], [123.0, 1200.0], [123.0, 1224.0], [35.0, 1224.0]], 'pred_id': 0, 'pred': 'O'}, {'transcription': '單分類', 'bbox': [39, 1264, 116, 1279], 'points': [[39.0, 1264.0], [116.0, 1264.0], [116.0, 1279.0], [39.0, 1279.0]], 'pred_id': 0, 'pred': 'O'}, {'transcription': '轉眼红包', 'bbox': [442, 1264, 519, 1279], 'points': [[442.0, 1264.0], [519.0, 1264.0], [519.0, 1279.0], [442.0, 1279.0]], 'pred_id': 0, 'pred': 'O'}]] 
        OUTPUT:
            "kie_ser" : {
                "header" : "15:00",
                "header_loc" : [ 175, 56, 264, 82 ],
                "time" : "王梁",
                "time_loc" : [ 502, 455, 578, 490 ],
                "time_trans" : "1789",
                "time_trans_loc" : [ 341, 966, 721, 1000 ],
            }
        '''
        # s1
        labels = {'HEADER': 0, 'QUESTION': 1, 'ANSWER': 2, 'TIME': 3, 'TIME_TRANS': 4, 'SERI': 5, 'MONEY': 6, 'BUTTON': 7, 'STK':8, 'CONFIRM': 9, 'NOTE1':10, 'NOTE2':11}
        labels.pop('QUESTION', None) 
        labels.pop('ANSWER', None)
        
        # s2
        ser_other = {k.lower(): None for k in labels}
        for key_value in ser_res_other[0]:
            txt = key_value['transcription']
            if key_value['pred'] in labels:
                bbox = key_value['bbox']
                bbox = fit_bbox_2(image, bbox)
                
                key = key_value['pred'].lower()
                ser_other[key] = txt
                ser_other[key + "_loc"] = bbox
        
        return ser_other