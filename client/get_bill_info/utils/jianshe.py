from .all_bank import get_info, strpos, get_info_left2right


def jianshe_info(text_by_line, bank_get_info, info: dict = {}):
    info["trans_time"] = get_info(text_by_line, bank_get_info["trans_time"][0])
    info["seri_number"] = get_info(text_by_line, bank_get_info["seri_number"][0])

    info["payee_acc"] = get_info(text_by_line, bank_get_info["payee_acc"][0])
    info["payee_name"] = get_info(text_by_line, bank_get_info["payee_name"][0])
    text_rm = text_by_line.replace(bank_get_info["payee_acc"][0], "", 1)
    text_rm = text_rm.replace(bank_get_info["payee_name"][0], "", 1)
    text_rm = text_rm.replace("专用章", "")
    text_rm = text_rm.replace("电子回单", "")
    text_rm = text_rm.replace("显示完整账号", "")
    info["payer_acc"] = get_info(text_rm, bank_get_info["payer_acc"][0])
    info["payer_name"] = get_info_stamp_cases(text_rm, bank_get_info["payer_name"][0])

    return info


def jianshe_info_v3(text_by_line, bank_get_info, info: dict = {}):
    for key in bank_get_info:
        if key == "payee_name":
            # Not have key value, get based on before line
            info[key] = get_payer_name_v3(text_by_line, bank_get_info[key][0])
        else:
            for v in bank_get_info[key]:
                info[key] = get_info(text_by_line, v)
                if info[key] is not None:
                    break

    return info


def get_info_stamp_cases(text_by_line, key):
    start_pos = strpos(text_by_line, key)
    if start_pos is None:
        return None
    else:
        len_ocrDetails = len(text_by_line)
        start_pos += len(key)
        if start_pos >= len_ocrDetails:
            return None

    if text_by_line[start_pos] == "|":
        start_pos -= len(key)
        return get_info_left2right(text_by_line, start_pos)

    start_pos += 1
    if start_pos == len_ocrDetails:
        return None

    position_end = start_pos + 1
    while text_by_line[position_end] != "|" and text_by_line[position_end] != " ":
        position_end += 1
        if position_end == len_ocrDetails:
            break

    return text_by_line[start_pos:position_end].strip()


def get_payer_name_v3(text_by_line, key):
    start_pos = strpos(text_by_line, key)
    if start_pos is None:
        return None
    else:
        len_ocrDetails = len(text_by_line)
        start_pos += len(key)
        if start_pos >= len_ocrDetails:
            return None

    while text_by_line[start_pos] != "|":
        start_pos += 1

    start_pos += 1
    if start_pos == len_ocrDetails:
        return None

    position_end = start_pos + 1
    while text_by_line[position_end] != "|" and text_by_line[position_end] != " ":
        position_end += 1
        if position_end == len_ocrDetails:
            break

    return text_by_line[start_pos:position_end].strip()
