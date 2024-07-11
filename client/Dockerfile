FROM python:3.10.13

RUN apt-get update
RUN pip install -U pip

# RUN apt update -y
RUN apt install -y software-properties-common

RUN apt install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libsqlite3-dev libreadline-dev \
     libffi-dev curl libbz2-dev pkg-config make
RUN apt-get install -y libgl1-mesa-dev

COPY ./client/requirements.txt /venv/requirements.txt
WORKDIR /venv
RUN python3 -m pip install -r requirements.txt
COPY ./client /venv

CMD ["python3", "api_model.py"]