import json
from kafka import KafkaProducer


class Producer:
    def __init__(self) -> None:
        self.producer = KafkaProducer(
            bootstrap_servers=[
                "16.163.245.213:9092",
                "16.163.245.213:9093",
                "16.163.245.213:9094",
            ],
            value_serializer=lambda v: json.dumps(v).encode(),
        )

    def send_data(self, content, topic) -> None:
        self.producer.send(topic=topic, value=content)


def send_result(kafkaP: Producer, alert_info, topic_alert):
    kafkaP.send_data(topic=topic_alert, content=alert_info)


def create_info_alert(item, code_error, time_pull_kafka: str = ""):
    res = {}

    res["author"] = "linhlee"
    res["url"] = item["url"]
    res["user_name"] = item["user_name"]
    res["site_id"] = item["site_id"]
    res["site_name"] = item["site_name"]
    res["order_amount"] = item["order_amount"]
    res["order_status"] = item["order_status"]

    bank_name = None
    if item["bank_name_cn"] is not None and item["bank_code"] is not None:
        bank_name = item["bank_name_cn"] + "V" + item["bank_code"][-2]
    res["bank_name"] = bank_name

    res["bank_name_cn"] = item["bank_name_cn"]
    res["bank_code"] = item["bank_code"]
    res["order_create_time"] = item["order_create_time"]
    res["order_confirm_time"] = item["order_confirm_time"]
    res["order_complete_time"] = item["order_complete_time"]
    res["pull_from_kafka_time"] = time_pull_kafka
    res["idx_url"] = item["idx_url"]
    res["group_name"] = item["group_name"]
    res["order_no"] = item["order_no"]
    res["device_no"] = item["device_no"]
    res["type"] = "detection_rule"
    res["error_code"] = code_error
    res["bill_info"] = item["bill_info"]
    res["member_grade"] = item["member_grade"]
    res["credit_rate"] = item["credit_rate"]

    res["time_id"] = item["time_id"]
    res["total_amount"] = item["total_amount"]
    res["total_bill"] = item["total_bill"]
    res["pay_type"] = item["pay_type"]
    res["actual_pay_type"] = item["actual_pay_type"]
    res["source_key"] = item["source_key"]

    return res
