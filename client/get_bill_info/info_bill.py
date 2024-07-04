import yaml

from utils.data_define import DataDefine

from get_bill_info.utils.all_bank import *


# def split_filed_info(cfg, info_dict: dict):
#     # letter field first, number field last

#     if info_dict["bank_code"] in cfg:
#         current_class = cfg[info_dict["bank_code"]][0]
#         future_class = cfg[info_dict["bank_code"]][1]  # only change number field
#         if info_dict[future_class]:
#             return

#         current_text: str = info_dict[current_class].get("text")
#         if current_text is not None:
#             rm_char_ls = [" ", "(", ")", "（", "）", ">"]
#             for char in rm_char_ls:
#                 current_text = current_text.replace(char, "")

#             for idx, char in enumerate(current_text):
#                 if char.isnumeric():
#                     info_dict[current_class]["text"] = current_text[:idx]

#                     info_dict[future_class] = {}
#                     info_dict[future_class]["text"] = current_text[idx:]
#                     info_dict[future_class]["points"] = info_dict[current_class][
#                         "points"
#                     ]
#                     info_dict[future_class]["conf"] = info_dict[current_class]["conf"]
#                     break

#     return


class InfoPostProcessing:
    def __init__(self) -> None:
        self.key_text_type = [
            "beneficiary_account_name_value",
            "payer_account_name_value",
        ]

        with open("cfg/bill_info/check_bank_info.yaml") as f:
            self.check_info = yaml.load(f, Loader=yaml.FullLoader)
            f.close()

        self.range_check = 3

    def check_text(self, info: dict, bank_code):
        ch_rm_all_bank = self.check_info["character_rm_all_bank"]
        for key in ch_rm_all_bank:
            if info.get(key) is not None:
                info[key] = remove_character(info[key], ch_rm_all_bank[key])

        ch_rm = self.check_info["character_rm"]
        if bank_code in ch_rm:
            for key in ch_rm[bank_code]:
                if info.get(key) is not None:
                    info[key] = remove_character(info[key], ch_rm[bank_code][key])

        check_all_number = self.check_info["check_all_number"]
        if bank_code in check_all_number:
            for key in check_all_number[bank_code]:
                if info.get(key) is not None:
                    info[key] = check_number(info[key])

        for key in self.key_text_type:
            if info.get(key) is not None:
                info[key] = check_full_text(info[key])

        if info.get("serial_number_value") is not None:
            for char in info["serial_number_value"]:
                if "\u4e00" <= char <= "\u9fff":
                    info["serial_number_value"] = None

        if info.get("transfer_time_value") is not None:
            info["transfer_time_value"] = format_time(info["transfer_time_value"])
        if info.get("phone_time") is not None:
            info["phone_time"] = format_phone_time(info["phone_time"])

        check_len_value_all = self.check_info["len_value_all_bank"]
        for key in check_len_value_all:
            if info.get(key) is not None:
                info[key] = check_len(info[key], check_len_value_all[key])

        check_len_value = self.check_info["len_value"]
        if bank_code in check_len_value:
            for key in check_len_value[bank_code]:
                if info.get(key) is not None:
                    info[key] = check_len(info[key], check_len_value[bank_code][key])

        return info

    # def __call__(self, model_res: list[dict], info: DataDefine) -> None:
    #     number_seri = 1
    #     for res in model_res:
    #         if res.get("pred") is None or res.get("pred") == "NONE":
    #             continue

    #         field = res["pred"].lower()

    #         if field == "transfer_money" and info.transfer_money:
    #             continue

    #         if (
    #             field == "serial_number_value"
    #             and info.serial_number_value  # check len dict > 0
    #             and info.bank_code in self.two_line_seri
    #             and number_seri == 1
    #         ):
    #             info.serial_number_value["text"] += res["transcription"]
    #             number_seri += 1

    #         if info.__dict__[field] and res["conf"] <= info.__dict__[field]["conf"]:
    #             continue

    #         setattr(
    #             info,
    #             field,
    #             {
    #                 "text": res["transcription"],
    #                 "points": res["points"],
    #                 "conf": res["conf"],
    #             },
    #         )

    #     split_filed_info(self.check_info["split_letter_number"], info.__dict__)
    #     self.check_text(info.__dict__)

    #     return

    def process(self, model_res: list[dict], bank_code: str) -> None:
        info_text = {}
        info_boxes = {}
        info_conf = {}

        for idx, res in enumerate(model_res):
            if res.get("pred") is None or res.get("pred") == "NONE":
                continue

            field = res["pred"].lower()

            if field in info_text:
                continue

            info_text[field] = res["transcription"]
            # two line
            for i in range(1, self.range_check):
                idx_2 = idx + i

                if idx_2 >= len(model_res):
                    break

                if "pred" not in model_res[idx_2]:
                    continue

                if model_res[idx_2]["pred"].lower() == field:
                    # Two lines of trans time are adjacent to each other
                    if model_res[idx_2]["transcription"] != info_text[field]:
                        info_text[field] += model_res[idx_2]["transcription"]

            info_conf[field] = res["conf"]

        return self.check_text(info_text, bank_code)
