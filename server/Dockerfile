# FROM nvcr.io/nvidia/tritonserver:23.10-py3
FROM nvcr.io/nvidia/tritonserver:24.05-py3-min

# RUN apt-get update
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip
RUN python3 -m pip install -U pip

# RUN apt update -y
# RUN apt install -y software-properties-common
# RUN add-apt-repository ppa:deadsnakes/ppa -y

RUN apt install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libsqlite3-dev libreadline-dev \
     libffi-dev curl libbz2-dev pkg-config make
RUN apt-get install -y libgl1-mesa-dev

# RUN python3 -m pip install paddlepaddle-gpu==2.5.2.post120 -f https://www.paddlepaddle.org.cn/whl/linux/cudnnin/stable.html
RUN python3 -m pip install paddlepaddle-gpu==2.6.1.post120 -f https://www.paddlepaddle.org.cn/whl/linux/cudnnin/stable.html
RUN python3 -m pip install paddleocr==2.7.0.3 -I blinker
RUN python3 -m pip install paddlenlp==2.8.1

COPY ./server/requirements.txt /venv/requirements.txt
WORKDIR /venv
RUN python3 -m pip install -r requirements.txt
COPY ckpts /venv/ckpts
COPY ./server /venv
RUN mv ./modeling.py /usr/local/lib/python3.10/dist-packages/paddlenlp/transformers/layoutxlm/modeling.py
RUN mv ./triton.py /usr/local/lib/python3.10/dist-packages/pytriton/triton.py

CMD ["python3", "triton_server.py"]