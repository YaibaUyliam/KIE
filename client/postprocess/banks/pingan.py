from .all_bank import remove_copy_text, strpos, get_info_left2right


def pingan_info_v2(text_by_line, bank_get_info, info: dict = {}):
    # ch_rm = [" 复制", " 　复制"]
    text_by_line = remove_copy_text(text_by_line)

    for key in bank_get_info:
        for v in bank_get_info[key]:
            # Can use to get name
            info[key] = get_serinumber_pingan(text_by_line, v)
            if info[key] is not None:
                break

    return info


def get_serinumber_pingan(text_by_line, key):
    text = ""

    while True:
        start_pos = strpos(text_by_line, key)
        if start_pos is None:
            return None
        else:
            if (
                text_by_line[start_pos - 1] == "|"
                and text_by_line[start_pos + len(key)] == " "
            ):
                break
            text_by_line = text_by_line[start_pos + len(key) :]

    start_pos += len(key)
    if text_by_line[start_pos] == "|":
        start_pos -= len(key)
        return get_info_left2right(text_by_line, start_pos)

    len_ocrDetails = len(text_by_line)
    start_pos += 1
    if start_pos == len_ocrDetails:
        return None

    position_end = start_pos + 1
    while text_by_line[position_end] != "|":
        position_end += 1
        if position_end == len_ocrDetails:
            break

    text += text_by_line[start_pos:position_end].strip()

    start_pos_2 = position_end + 1
    position_end_2 = start_pos_2 + 1
    if position_end_2 >= len_ocrDetails:
        return text

    while text_by_line[position_end_2] != "|":
        if not text_by_line[position_end_2].isnumeric():
            return text

        position_end_2 += 1
        if position_end_2 == len_ocrDetails:
            break

    text += text_by_line[start_pos_2:position_end_2].strip()

    return text
