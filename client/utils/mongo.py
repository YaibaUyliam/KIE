import logging
from datetime import datetime

from pymongo import MongoClient
from yaml import safe_load


class Mongo:
    def __init__(
        self,
        cfg: dict = None,
        enable_signal: bool = True,
    ):
        # if enable_signal:
        #     signal.signal(signal.SIGINT, self.join)
        #     signal.signal(signal.SIGTERM, self.join)
        self.__db = MongoClient(cfg["connection_string"])[cfg["database"]]
        # self.__thread_pool = ThreadPool()

    # def join(self):
    #     self.__thread_pool.close()
    #     self.__thread_pool.join()

    def insert_one(self, collection: str, document: dict):
        try:
            self.__db[collection].insert_one(document)
        except:
            logging.exception(document)

    # def insert_one(self, collection: str, document: dict):
    #     self.__thread_pool.apply_async(self.__insert_one, args=(collection, document))

    def find(
        self,
        collection: str,
        filter: dict = {},
        projection: dict = {},
        sort: list = [],
        limit=0,
        **kwargs
    ):
        return self.__db[collection].find(
            filter=filter, projection=projection, sort=sort, limit=limit, **kwargs
        )


class OrderDB:
    def __init__(self, config_path: str = None):
        if config_path:
            with open(config_path) as f:
                config = safe_load(f)

        client = MongoClient(config["OrderDB"]["connection_string"])
        db = client["cms"]
        self.collection = db["orders-v2"]

        self.sort_order = [("_id", -1)]
        self.projection = {
            "_id": 0,
            "auditStatusEn": 1,
            "clientIp": 1,
            "clientIpLocation": 1,
            "username": 1,
        }

    def query(self, device_no: str, order_create_time: str) -> list[dict]:
        query = {
            "deviceNo": device_no,
            "orderCreateTime": {"$lt": order_create_time},
            "memberGrade": {"$in": ["VIP0", "0"]},
        }

        return (
            self.collection.find(query, self.projection)
            .sort(self.sort_order)
            .limit(100)
        )

    def is_fake(self, item: dict):
        if (
            item["device_no"] is None
            or len(item["device_no"]) == 0
            or item["order_create_time"] is None
            or len(item["order_create_time"]) == 0
            or item["member_grade"] not in ["VIP0", "0"]
        ):
            return False

        res_list = self.query(item["device_no"], item["order_create_time"])

        client_ip_loc_list = []
        user_list = []
        for res in res_list:
            if (
                res.get("clientIpLocation") is not None
                and res.get("username") is not None
            ):
                ip_loc_split = res["clientIpLocation"].strip().split("|")
                client_ip_loc_list.append(ip_loc_split[1])
                user_list.append(res["username"])

        if len(set(client_ip_loc_list)) > 2 and len(set(user_list)) > 1:
            logging.info(item["order_create_time"])
            logging.info(item["order_no"])
            logging.info(item["device_no"])
            logging.info(client_ip_loc_list)
            return True

        return False


class ModelResultDB:
    def __init__(self, config_path: str = None):
        if config_path:
            with open(config_path) as f:
                config = safe_load(f)

        client = MongoClient(config["ModelResultDB"]["connection_string"])
        db = client["ai-team"]
        # self.collection = db["model-result"]
        self.collection = db["classify_ocr"]

        self.sort_order = [("_id", -1)]
        self.projection = {
            "_id": 0,
            # "error_code": 1,
            # "order_create_time": 1,
            "order_no": 1,
        }

    def query(self, user_name: str) -> list[dict]:
        query = {"user_name": user_name}

        return (
            self.collection.find(query, self.projection).sort(self.sort_order).limit(10)
        )

    def is_used_phone(self, user_name, error_code, order_no, order_create_time) -> bool:
        user_history_list = self.query(user_name)
        for user_history in user_history_list:
            logging.info(user_history)
            if (
                error_code in user_history["error_code"]
                and order_no != user_history["order_no"]
            ):
                order_history_time = user_history["order_create_time"]
                if order_create_time is not None and order_history_time is not None:
                    time1 = datetime.strptime(order_create_time, "%Y-%m-%d %H:%M:%S")
                    time2 = datetime.strptime(order_history_time, "%Y-%m-%d %H:%M:%S")

                    if time2 > time1:
                        continue

                    if (time1 - time2).total_seconds() < 10:
                        return False

                return True

        return False

    def is_first_time(self, user_name, order_no):
        count = 0
        user_history_list = self.query(user_name)

        for user_history in user_history_list:
            logging.info(user_history)
            if user_history["order_no"] != order_no:
                count += 1
                break

        if count == 0:
            return True

        return False


if __name__ == "__main__":
    t = ModelResultDB("/home/yaiba/project/KIE/client/cfg/mongo.yaml")
    res = t.query("a")
    print(len(list(res)))
