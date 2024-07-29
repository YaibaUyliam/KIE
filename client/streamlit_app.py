import streamlit as st
import requests
import base64
import json
import os

from pymongo import MongoClient
from dotenv import load_dotenv


load_dotenv()  # By default, load_dotenv doesn't override existing environment variables.


client = MongoClient(os.environ.get("URL_DB"), serverSelectionTimeoutMS=5000)
db = client["ai-team"]
collection = db["classify_ocr"]
projection = {"url": 1, "_id": 0}
sort_order = [("_id", -1)]
limit = 20

st.set_page_config(layout="wide")
st.title("Key Info Extraction")

# num_rows = st.slider("Number of rows", min_value=1, max_value=10)

url = st.text_input("Nhập URL của bạn vào đây")
bank_code = st.text_input("Nhập bank code của bạn vào đây")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])


if uploaded_file or url or bank_code:
    headers = {}
    files = {}

    if uploaded_file:
        files = [
            (
                "file",
                (
                    "cq02ts-b77d2310b0ad06-151349.jpg",
                    uploaded_file,
                    "image/jpeg",
                ),
            )
        ]
        order_list = [{}]

    elif url:
        headers = {"Content-Type": "application/json"}
        order_list = [{"url": url}]

    else:
        headers = {"Content-Type": "application/json"}

        query = {"bank_code": bank_code.strip()}
        order_list = list(
            collection.find(query, projection).sort(sort_order).limit(limit)
        )

    if st.button("Process Image"):
        for payload in order_list:
            col1, col2, col3 = st.columns(3)

            payload = json.dumps(payload)

            response = requests.request(
                "POST",
                "http://172.19.30.25:5003/ser_re_visual",
                headers=headers,
                data=payload,
                files=files,
            )

            if response.status_code == 200:
                data = response.json()

                result_1_b64 = data["img_ser"]
                result_2_b64 = data["img_re"]
                result_3 = data["img_ser_post"]

                result_1, result_2 = None, None
                if result_1_b64:
                    result_1 = base64.b64decode(result_1_b64.split(",")[1])
                if result_2_b64:
                    result_2 = base64.b64decode(result_2_b64.split(",")[1])

                with st.container():
                    with col1:
                        if result_2:
                            st.image(result_2, caption="Result RE", use_column_width=True)  # fmt: skip
                        else:
                            pass
                    with col2:
                        st.image(result_1, caption="Result SER", use_column_width=True)
                    with col3:
                        st.write(result_3)

            else:
                st.error("Failed to process image")
