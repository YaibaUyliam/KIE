import json
import yaml
import logging
import traceback
import os
import time

import signal
from multiprocessing.pool import ThreadPool
from threading import Event, Lock

from kafka import KafkaConsumer

from utils.data_define import DataDefine
from triton_client import KieClient  # , YoloClient


test_env = os.environ["TEST_ENV"].lower() == "true"

if test_env is False:
    filename = "log/record_bill_info.log"
else:
    filename = "log/record_bill_info_test.log"

logging.basicConfig(
    filename=filename,
    level=logging.INFO,
    format=f"%(asctime)s %(levelname)s %(name)s : %(message)s",
)
logger = logging.getLogger(__name__)


class BillInfo:
    def __init__(self):
        logging.info("Starting ....")

        self.process = 5
        self.pool = ThreadPool(self.process)
        self.lock = Lock()
        self.stop_event = Event()

        self.consumer = KafkaConsumer(
            bootstrap_servers=[
                "16.163.245.213:9092",
                "16.163.245.213:9093",
                "16.163.245.213:9094",
            ],
            auto_offset_reset="earliest",
            group_id="kie-test",
            value_deserializer=lambda m: json.loads(m),
        )

        self.consumer.subscribe(["classify_ocr"])

        with open("cfg/bank_get_info.yaml") as f:
            self.bank_run = yaml.load(f, Loader=yaml.FullLoader)

        self.client_kie = KieClient(config_path="cfg/mongo.yaml")

    def run(self):
        while True:
            try:
                with self.lock:
                    data = self.consumer.poll(timeout_ms=1000, max_records=1)

                for _, items in data.items():
                    for item in items:
                        item = item.value
                        info = DataDefine(item)

                        # if item["bank_code"] not in self.bank_run:
                        #     continue

                        if info.img_nd is not None:
                            self.client_kie(info)

                if self.stop_event.is_set():
                    return

            except Exception as e:
                logging.error(e)
                logging.error(type(e).__name__)
                logging.error(traceback.format_exc())
                logging.error(item["url"])
                logging.error(item["order_no"])

    def start(self):
        for _ in range(self.process):
            self.pool.apply_async(func=self.run)

        self.stop_event.wait()
        # self.pool.close()
        # self.pool.join()


def signal_handler(sig, frame):
    print("Ctrl+C received ...")

    bi.stop_event.set()
    bi.pool.close()
    bi.pool.join()

    print("All threads are done. Exiting.")
    logging.info("All threads are done. Exiting.")


if __name__ == "__main__":
    bi = BillInfo()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    bi.start()
