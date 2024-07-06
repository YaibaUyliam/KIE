from .all_bank import get_info, strpos


def youzheng_info(text_by_line, bank_get_info, info: dict = {}):
    for key in bank_get_info:
        if key == "payee_name":
            info[key] = get_info(text_by_line[30:], bank_get_info[key][0])
        elif key == "payer_name":
            info[key] = get_payer_name(text_by_line, bank_get_info[key][0])
        else:
            for v in bank_get_info[key]:
                info[key] = get_info(text_by_line, v)
                if info[key] is not None:
                    break

    return info


def get_payer_name(text_by_line, key):
    pos_payer_name = strpos(text_by_line, key)
    if pos_payer_name is None:
        return None
    
    end = pos_payer_name
    start = pos_payer_name - 1
    while text_by_line[start] != "|":
        start -= 1
        if start == 0:
            return None

    return text_by_line[start + 1 : end]
