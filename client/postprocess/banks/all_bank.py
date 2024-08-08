def strpos(ocrDetails: str = "", text: str = ""):
    if ocrDetails is None:
        return None

    if text in ocrDetails:
        return ocrDetails.index(text)

    return None


def str_contains(ocrDetails: str = "", text: str = "") -> bool:
    if text in ocrDetails:
        return True

    return False


def get_info(text_by_line, key, signal_stop="|"):
    start_pos = strpos(text_by_line, key)
    if start_pos is None:
        return None
    else:
        len_ocrDetails = len(text_by_line)
        start_pos += len(key)
        if start_pos >= len_ocrDetails:
            return None

    # while not text_by_line[start_pos].isnumeric():
    # Case value have character "***"
    if text_by_line[start_pos] == signal_stop:
        start_pos -= len(key)
        return get_info_left2right(text_by_line, start_pos)

    start_pos += 1
    if start_pos >= len_ocrDetails - 1:
        return None

    # if get_pos_end is True:
    position_end = start_pos + 1
    while text_by_line[position_end] != "|":
        position_end += 1
        if position_end == len_ocrDetails:
            break

    return text_by_line[start_pos:position_end].strip()


def get_info_left2right(text_by_line, end_pos, signal_stop="|"):
    # while not text_by_line[end_pos - 1].isnumeric():
    end_pos -= 1
    if end_pos <= 0:
        return None

    start_pos = end_pos - 1
    while text_by_line[start_pos - 1] != signal_stop:
        start_pos -= 1
        if start_pos <= 0:
            return None

    return text_by_line[start_pos:end_pos].strip()


def get_info_two_line(text_by_line, key):
    text = ""

    start_pos = strpos(text_by_line, key)
    if start_pos is None:
        return None
    else:
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


def find_number(text: str):
    for idx, char in enumerate(text):
        if char.isnumeric():
            return idx

    return None


# def check_number(text, check_time=False):
#     """
#     Check all word is number or not, except * chacracter
#     Example: '21345**1231' is '21345**1231'
#              '21312213 abc' is '21312213'
#     """
#     chars_to_remove = ".:/-+"
#     text_short = "".join(char for char in text if char not in chars_to_remove)
#     text_short = text_short.replace(" ", "")

#     if text_short.isnumeric() or text_short.replace("*", "").isnumeric():
#         if check_time is False:
#             return text_short
#         else:
#             return text

#     idx_start = find_number(text_short)
#     idx_end = find_number(text_short[::-1])
#     if idx_end is None or idx_start + idx_end <= len(text_short):
#         return None
#     idx_end = -idx_end
#     if idx_end == 0:
#         text_true = text_short[idx_start:]
#     else:
#         text_true = text_short[idx_start:idx_end]

#     if text_true.isnumeric():
#         return text_true

#     return None


def check_number(text):
    """
    Check all word is number or not, except * chacracter
    Example: '21345**1231' is '21345**1231'
             '21312213 abc' is '21312213'
    """
    chars_to_remove = ".:/-+*"
    text_short = "".join(char for char in text if char not in chars_to_remove)
    text_short = text_short.replace(" ", "")

    if text_short.isnumeric():
        return text


def check_full_alpha(text: str):
    chars_to_remove = ".:/-"
    text_short = "".join(char for char in text if char not in chars_to_remove)
    text_short = text_short.replace(" ", "")

    for w in text:
        if w.isnumeric():
            return None

    return text


def format_acc(text: str):
    end_str_1 = text.find("*")
    start_str_2 = text[::-1].find("*")
    if end_str_1 == -1 or start_str_2 == -1:
        return text

    return text[:end_str_1] + "**" + text[-start_str_2:]


def check_len(text, lenght=0):
    if len(text) < lenght:
        return None

    return text


def remove_copy_text(text_by_line):
    # 指令号 30758400799820231107835 5复制
    len_text_rm = 2
    pos_copy_text = strpos(text_by_line, "复制")
    while pos_copy_text is not None:
        while (
            text_by_line[pos_copy_text - 1] != " "
            or text_by_line[pos_copy_text - 2] == " "
        ):
            pos_copy_text -= 1
            len_text_rm += 1
            if text_by_line[pos_copy_text] == "|":
                break

        text_by_line = text_by_line.replace(
            text_by_line[pos_copy_text : pos_copy_text + len_text_rm], ""
        )
        pos_copy_text = strpos(text_by_line, "复制")

    return text_by_line


def remove_character(text: str, ch_rm_list):
    for ch_rm in ch_rm_list:
        text = text.replace(ch_rm, "")

    return text.replace(" ", "")


def check_full_text(text: str):
    chars_to_remove = ".:/-*"
    text_short = "".join(char for char in text if char not in chars_to_remove)
    text_short = text_short.replace(" ", "")

    for w in text:
        if w.isnumeric():
            return None

    return text


def format_time(text: str):
    text = text.replace(" ", "")
    text = text.replace("：", ":")

    # 2024年8月2日01:17:27 2024年7月17日 10:37:06
    if "日" in text:
        day_index = text.find("日") + 1
        if day_index > 0 and day_index < len(text) and text[day_index] != " ":
            text = text[:day_index] + " " + text[day_index:]

    # 2024年01月14
    elif (
        len(text) > 11
        and text[10] != " "
        and text[10] != "日"  # 2024年04月03日09:01:56
    ):
        text = text[:10] + " " + text[10:]

    if len(text) not in [8, 10, 11, 16, 18, 19]:
        return None

    if "/" in text:
        text_list = list(text)
        text_list[4] = "/"
        text_list[7] = "/"
        text = "".join(text_list)

    return text.strip()


def format_phone_time(text: str):
    text = text.replace(" ", "")

    if len(text) not in [5, 7]:
        text = None

    return text
