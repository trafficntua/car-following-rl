FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
WORKDIR /project

RUN apt update && apt upgrade -y && apt install -y software-properties-common && add-apt-repository ppa:sumo/stable && apt update && DEBIAN_FRONTEND=noninteractive apt install -y sumo sumo-tools sumo-doc
RUN echo "export SUMO_HOME=/usr/share/sumo" >> ~/.bashrc
RUN pip install --upgrade pip
RUN pip install jupyterlab
COPY ./project /project
