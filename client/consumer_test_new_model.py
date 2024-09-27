import json
import yaml
import logging
import traceback
import os
from dotenv import load_dotenv

import signal
from multiprocessing.pool import ThreadPool
from threading import Event, Lock

from kafka import KafkaConsumer

from utils.data_define import DataDefine
from triton_client import KieClient  # , YoloClient


config = load_dotenv()
test_env = os.environ["TEST_ENV"].lower() == "true"

# if test_env is False:
#     filename = "log/record_bill_info.log"
# else:
#     filename = "log/record_bill_info_test.log"

logging.basicConfig(
    # filename=filename,
    level=logging.INFO,
    format=f"%(asctime)s %(levelname)s %(name)s : %(message)s",
)
logger = logging.getLogger(__name__)


class BillInfo:
    def __init__(self):
        logger.info("Starting ....")

        self.process = int(os.environ["PROCESS"])
        self.pool = ThreadPool(self.process)
        self.lock = Lock()
        self.stop_event = Event()

        self.consumer = KafkaConsumer(
            bootstrap_servers=os.environ["KAFKA"].split(","),
            auto_offset_reset="earliest",
            group_id=os.environ["GROUP_ID"],
            value_deserializer=lambda m: json.loads(m),
        )

        self.consumer.subscribe(["classify_ocr"])

        connection_db = os.environ["URL_DB"]
        # with open("cfg/bank_get_info.yaml") as f:
        #     self.bank_run = yaml.load(f, Loader=yaml.FullLoader)

        self.client_kie = KieClient(connection_db)

    def run(self):
        while True:
            try:
                with self.lock:
                    data = self.consumer.poll(timeout_ms=2000, max_records=1)

                for _, items in data.items():
                    for item in items:
                        item = item.value
                        if item["bank_code"] == "C000":
                            continue

                        info = DataDefine(item)

                        if info.img_nd is not None:
                            self.client_kie(info)

                if self.stop_event.is_set():
                    return

            except Exception as e:
                logger.error(e)
                logger.error(type(e).__name__)
                logger.error(traceback.format_exc())
                logger.error(item["url"])
                logger.error(item["order_no"])

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
    logger.info("All threads are done. Exiting.")


if __name__ == "__main__":
    bi = BillInfo()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    bi.start()
