from .all_bank import (
    get_info,
    remove_copy_text,
    get_info_two_line,
)


def jiaotong_info_v1(text_by_line: str, bank_get_info, info: dict = {}):
    pos_payer_acc = None

    for v in bank_get_info["seri_number"]:
        info["seri_number"] = get_info(text_by_line, v)

        if info["seri_number"] is not None:
            info["seri_number"] = info["seri_number"].replace(" ", "")
            info["seri_number"] = info["seri_number"][:25]
            break

    for v in bank_get_info["payer_name"]:
        list_pos = [pos for pos, char in enumerate(text_by_line) if char == v]
        text_short = text_by_line.replace("交通银行", " ")
        for pos_key in list_pos:
            if text_short[pos_key - 1] == "|":
                if text_short[pos_key + 1] == "|":
                    info["payer_name"] = get_info(text_short, v, signal_stop=" ")
                else:
                    info["payer_name"] = get_info(text_short, v)

                if info["payer_name"] is not None:
                    pos_payer_acc = pos_key
                    break

            else:
                text_short = text_short.replace(v, " ", 1)

    for v in bank_get_info["payer_acc"]:
        info["payer_acc"] = get_info(text_by_line[pos_payer_acc:], v, signal_stop=" ")
        if info["payer_acc"] is not None:
            break

    return info


def jiaotong_info_v2(text_by_line, bank_get_info, info: dict = {}):
    for key in bank_get_info:
        if key == "seri_number":
            text_rm = remove_copy_text(text_by_line)
            for v in bank_get_info[key]:
                info[key] = get_info_two_line(text_rm, v)

                if info[key] is not None:
                    break

    return info
