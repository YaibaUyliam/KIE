import streamlit as st
import requests
import base64
import json


st.set_page_config(layout="wide")
st.title("Key Info Extraction")

col1, col2, col3 = st.columns(3)

url = st.text_input("Nhập URL của bạn vào đây")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None or url is not None:
    headers = {}
    files = {}
    payload = {}

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

    else:
        headers = {"Content-Type": "application/json"}
        payload = json.dumps({"url": url})
    # b64_data = base64.b64encode(bytes_data).decode("utf-8")
    # image_data = f"data:image/jpeg;base64,{b64_data}"
    # with col1:
    #     st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

    if st.button("Process Image"):
        # if uploaded_file:
        response = requests.request(
            "POST",
            "http://127.0.0.1:5003/ser_re_visual",
            headers=headers,
            data=payload,
            files=files,
        )
        # else:
        #     response = requests.request(
        #         "POST", url, headers=headers, data=payload, files=files
        #     )

        if response.status_code == 200:
            data = response.json()
            result_1_b64 = data["img_ser"]
            result_2_b64 = data["img_re"]
            result_3 = data["img_ser_post"]
            result_1 = base64.b64decode(result_1_b64.split(",")[1])
            result_2 = base64.b64decode(result_2_b64.split(",")[1])

            with col1:
                st.image(result_2, caption="Result RE", use_column_width=True)
            with col2:
                st.image(result_1, caption="Result SER", use_column_width=True)
            with col3:
                st.write(result_3)
        else:
            st.error("Failed to process image")
