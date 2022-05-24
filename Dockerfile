# syntax=docker/dockerfile:1

FROM nvidia/cuda:10.2-base
FROM python:3.8.12-buster

RUN apt-get update && apt-get install -y --no-install-recommends \
     libglib2.0-0 libsm6 libxrender1 libxext6 libgl1 && \
     rm -rf /var/lib/apt/lists/*

WORKDIR /home

COPY requirements.txt requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Minimize image size 
RUN (apt-get autoremove -y; \
     apt-get autoclean -y) 

ENTRYPOINT [ "bash" ]
