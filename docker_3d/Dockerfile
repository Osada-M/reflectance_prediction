FROM nvcr.io/nvidia/tensorflow:21.09-tf2-py3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get -y install \
    gosu \
    vim \
    graphviz \
    v4l-utils \
    ffmpeg \
    wget \
    libssl-dev \
    libffi-dev \
    openssl \
    lsb-release \
    gnupg2 \
    libassimp-dev \
    && rm -rf /var/lib/apt/lists/*

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]


RUN wget https://www.python.org/ftp/python/3.9.18/Python-3.9.18.tar.xz \
    && tar -xf Python-3.9.18.tar.xz \
    && cd Python-3.9.18 \
    && ./configure --enable-optimizations \
    && make -j 4 \
    && make altinstall


RUN apt-get update && apt-get install -y
RUN apt-get install -y libgl1-mesa-dev
RUN apt-get autoremove -y
RUN python3.9 -m pip install --upgrade pip

RUN python3.9 -m pip install numpy
RUN python3.9 -m pip install tensorflow_model_optimization
RUN python3.9 -m pip install tensorflow
RUN python3.9 -m pip install --default-timeout=900 torch
RUN python3.9 -m pip install torchvision
RUN python3.9 -m pip install torchaudio
RUN python3.9 -m pip install scipy
RUN python3.9 -m pip install opencv-python
RUN python3.9 -m pip install matplotlib
RUN python3.9 -m pip install colorama
RUN python3.9 -m pip install pillow
RUN python3.9 -m pip install --default-timeout=900 scikit-learn
RUN python3.9 -m pip install pydot
RUN python3.9 -m pip install tqdm
RUN python3.9 -m pip install scikit-image
RUN python3.9 -m pip install openpyxl
RUN python3.9 -m pip install transformers
RUN python3.9 -m pip install tensorflow_probability
RUN python3.9 -m pip install ultralytics==8.0.120
RUN python3.9 -m pip install rtree
RUN python3.9 -m pip install open3d
RUN python3.9 -m pip install pyassimp

RUN python3.9 -m pip install osada
RUN python3.9 -m pip install git+https://github.com/tensorflow/examples.git
RUN python3.9 -m pip install git+https://github.com/openai/CLIP.git


ENV DEBIAN_FRONTEND=dialog
USER root
