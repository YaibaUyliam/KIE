from .all_bank import get_info, get_info_two_line, strpos


def zhongguo_info(text_by_line, bank_get_info, info: dict = {}):
    ch_rm = [" M", " O"]
    for ch in ch_rm:
        text_by_line = text_by_line.replace(ch, "")

    for key in bank_get_info:
        if key == "payee_acc" or key == "payer_acc":
            if "***" not in text_by_line:
                info[key] = get_info_two_line(text_by_line, bank_get_info[key][0])
                if info[key] is None:
                    info[key] = get_info(text_by_line, bank_get_info[key][0])
            else:
                info[key] = get_info(text_by_line, bank_get_info[key][0])
        else:
            for v in bank_get_info[key]:
                info[key] = get_info(text_by_line, v)
                if info[key] is not None:
                    break

    return info
