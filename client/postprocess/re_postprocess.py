from .util import fit_bbox_2

class REPostProcessing:
    def __init__(self) -> None:
        pass
    
    def check_same_bbox(self, bbox1, bbox2):
        return bbox1[0] == bbox2[0] and bbox1[1] == bbox2[1] and bbox1[2] == bbox2[2] and bbox1[3] == bbox2[3]

    def __call__(self, re_res, image):
        '''
            --> s1: couple
            {'id': None, 'key': ['服务费', [63, 808, 188, 860]], 'value': ['0.60', [335, 815, 417, 852]]}
            {'id': None, 'key': ['创建时间', [68, 902, 219, 947]], 'value': ['2024-07-0119:09:42', [338, 908, 682, 942]]}
            {'id': None, 'key': ['付款方式', [65, 995, 222, 1040]], 'value': ['账户余额>', [335, 995, 523, 1040]]}
            {'id': None, 'key': ['转账到', [65, 1088, 183, 1132]], 'value': ['中国邮政储蓄银行（6446）董玉龙', [340, 1088, 911, 1130]]}
            {'id': None, 'key': ['处理进度', [65, 1180, 222, 1225]], 'value': ['提交银行处理', [403, 1180, 636, 1225]]}
            {'id': None, 'key': ['处理进度', [65, 1180, 222, 1225]], 'value': ['07-0119:09', [403, 1243, 595, 1285]]}

            --> s2: cluster
            {'id': 0, 'key': ['服务费', [63, 808, 188, 860]], 'value': ['0.60', [335, 815, 417, 852]]}
            {'id': 1, 'key': ['创建时间', [68, 902, 219, 947]], 'value': ['2024-07-0119:09:42', [338, 908, 682, 942]]}
            {'id': 2, 'key': ['付款方式', [65, 995, 222, 1040]], 'value': ['账户余额>', [335, 995, 523, 1040]]}
            {'id': 3, 'key': ['转账到', [65, 1088, 183, 1132]], 'value': ['中国邮政储蓄银行（6446）董玉龙', [340, 1088, 911, 1130]]}
            {'id': 4, 'key': ['处理进度', [65, 1180, 222, 1225]], 'value': ['提交银行处理', [403, 1180, 636, 1225]]}
            {'id': 4, 'key': ['处理进度', [65, 1180, 222, 1225]], 'value': ['07-0119:09', [403, 1243, 595, 1285]]}
            {'id': 4, 'key': ['处理进度', [65, 1180, 222, 1225]], 'value': ['银行处理中', [405, 1332, 598, 1378]]}

            --> s3: merge
            {'key': ['处理进度', [65, 1180, 222, 1225]], 'value': [
                                                    ['提交银行处理', [403, 1180, 636, 1225]],
                                                    ['07-0119:09', [403, 1243, 595, 1285]],
                                                    ['银行处理中', [405, 1332, 598, 1378]],
                                                    ['07-0119:09', [405, 1400, 593, 1435]],
                                                    ['银行入账成功', [403, 1488, 632, 1530]],
                                                ]
                                            }
        '''
        # s1
        couples = []
        for key_value in re_res[0]:
            couple = {'id': None,'key': None, 'value': None}
            for element in key_value:
                txt = element['transcription']
                bbox = element['bbox']
                bbox = fit_bbox_2(image, bbox, type="xyxy")
                if element['pred'] == 'QUESTION':
                    couple['key'] = [txt, bbox]
                elif element['pred'] == 'ANSWER':
                    couple['value']= [txt, bbox]
            couples.append(couple)

        # s2
        couples[0]['id'] = 0
        for i in range(1, len(couples)):
            curr_couple = couples[i]
            prev_couple = couples[i-1]
            txt, bbox = curr_couple['key']
            if txt == prev_couple['key'][0] and self.check_same_bbox(bbox, prev_couple['key'][1]):
                couples[i]['id'] = prev_couple['id']
            else:
                couples[i]['id'] = prev_couple['id']+1

        # s3
        prev_id = -1
        new_couples = []
        for couple in couples:
            if couple['id'] != prev_id:
                new_couples.append({'key': [couple['key']], 'value': [couple['value']]})
                prev_id = couple['id']
            else:
                new_couples[-1]['value'] += [couple['value']]

        return new_couples