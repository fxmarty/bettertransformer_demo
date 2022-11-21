FROM ubuntu:22.04

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

RUN conda init bash

RUN pip install torch torchvision torchaudio

RUN pip install torchserve torch-model-archiver torch-workflow-archiver

RUN pip install transformers optimum

# install git
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    curl git && \
    apt-get clean

# install torchserve dependencies
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    openjdk-17-jdk && \
    apt-get clean

RUN curl -sL https://deb.nodesource.com/setup_14.x | bash -
RUN apt install -y nodejs

RUN pip install captum nvgpu

WORKDIR /workspace/bettertransformer_demo
# using wget because git clone would download tensorflow, etc. and we don't care
RUN wget -P ./distilbert-base-uncased-finetuned-sst-2-english/ https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/pytorch_model.bin
RUN wget -P ./distilbert-base-uncased-finetuned-sst-2-english/ https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/config.json

ADD "https://www.random.org/cgi-bin/randbyte?nbytes=10&format=h" skipcache
RUN git init . && git remote add -f origin https://github.com/fxmarty/bettertransformer_demo.git && git checkout main

ARG PROP_PATH
ARG MAR_NAME

ENV PROP_PATH_VAR=$PROP_PATH
ENV MAR_NAME_VAR=$MAR_NAME

RUN torch-model-archiver --model-name ${MAR_NAME_VAR} \
    --version 1.0 --serialized-file distilbert-base-uncased-finetuned-sst-2-english/pytorch_model.bin \
    --handler ./transformer_text_classification_handler.py \
    --extra-files "distilbert-base-uncased-finetuned-sst-2-english/config.json,./setup_config.json,./index_to_name.json" \
    -f \
    --export-path model_store

CMD torchserve --start --model-store model_store --models my_tc=${MAR_NAME_VAR}.mar --ncs --ts-config ${PROP_PATH_VAR} && tail -f /dev/null
