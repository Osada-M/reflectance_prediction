# FROM python:3.11.3-buster
# FROM python:3.10.11-buster
# FROM nvidia/cuda:11.3.1-base-ubuntu20.04

FROM nvcr.io/nvidia/tensorflow:21.09-tf2-py3
# FROM python:3.10

RUN apt-get update && apt-get upgrade -y && apt-get -y install gosu vim graphviz v4l-utils ffmpeg
# RUN apt-get -y install locales && localedef -f UTF-8 -i ja_JP ja_JP.UTF-8

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# ENV IMG_NAME=11.6.0-devel-ubuntu20.04 JAXLIB_VERSION=0.3.0

USER root

RUN apt-get install -y libgl1-mesa-dev
RUN apt-get autoremove -y
RUN pip install --upgrade pip

RUN pip install numpy==1.19.2
RUN pip install tensorflow_model_optimization
RUN pip install tensorflow==2.10.0
# RUN pip install tensorflow==2.12.0
RUN pip install --default-timeout=900 torch
RUN pip install torchvision
RUN pip install torchaudio
RUN pip install jax jaxlib
RUN pip install scipy
RUN pip install six
RUN pip install wheel
RUN pip install opencv-python
RUN pip install matplotlib
RUN pip install colorama
RUN pip install pillow
RUN pip install --default-timeout=900 scikit-learn
RUN pip install pandas
RUN pip install pydot
RUN pip install tqdm
RUN pip install line-bot-sdk
RUN pip install umap-learn
RUN pip install scikit-image
RUN pip install openpyxl
RUN pip install flax
RUN pip install ml_collections
RUN pip install einops
RUN pip install clu
RUN pip install ott ott-jax
RUN pip install ftfy
RUN pip install regex
RUN pip install gin-config
RUN pip install pyan
RUN pip install transformers
RUN pip install tensorflow_probability
RUN pip install ultralytics

RUN pip install osada
RUN pip install git+https://github.com/tensorflow/examples.git
RUN pip install git+https://github.com/openai/CLIP.git
RUN pip install git+https://github.com/Lednik7/CLIP-ONNX.git
RUN pip install onnxruntime-gpu

RUN pip install protobuf==3.20.0
RUN pip install torch --upgrade
