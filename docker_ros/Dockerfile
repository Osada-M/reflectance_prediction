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
    && rm -rf /var/lib/apt/lists/*

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]


RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' \
    && apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

RUN apt-get update && apt-get install -y \
    ros-noetic-desktop-full \
    && rm -rf /var/lib/apt/lists/*

RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
ENV ROS_PACKAGE_PATH /opt/ros/noetic/share


RUN wget https://www.python.org/ftp/python/3.10.6/Python-3.10.6.tar.xz \
    && tar -xf Python-3.10.6.tar.xz \
    && cd Python-3.10.6 \
    && ./configure --enable-optimizations \
    && make -j 4 \
    && make altinstall


RUN apt-get install -y libgl1-mesa-dev
RUN apt-get autoremove -y
RUN python3.10 -m pip install --upgrade pip

RUN python3.10 -m pip install numpy
RUN python3.10 -m pip install tensorflow_model_optimization
RUN python3.10 -m pip install tensorflow
RUN python3.10 -m pip install --default-timeout=900 torch
RUN python3.10 -m pip install torchvision
RUN python3.10 -m pip install torchaudio
RUN python3.10 -m pip install jax jaxlib
RUN python3.10 -m pip install jax==
RUN python3.10 -m pip install scipy
RUN python3.10 -m pip install six
RUN python3.10 -m pip install wheel
RUN python3.10 -m pip install opencv-python
RUN python3.10 -m pip install matplotlib
RUN python3.10 -m pip install colorama
RUN python3.10 -m pip install pillow
RUN python3.10 -m pip install --default-timeout=900 scikit-learn
RUN python3.10 -m pip install pandas
RUN python3.10 -m pip install pydot
RUN python3.10 -m pip install tqdm
RUN python3.10 -m pip install line-bot-sdk
RUN python3.10 -m pip install umap-learn
RUN python3.10 -m pip install scikit-image
RUN python3.10 -m pip install openpyxl
RUN python3.10 -m pip install flax
RUN python3.10 -m pip install ml_collections
RUN python3.10 -m pip install einops
RUN python3.10 -m pip install clu
RUN python3.10 -m pip install ott ott-jax
RUN python3.10 -m pip install ftfy
RUN python3.10 -m pip install regex
RUN python3.10 -m pip install gin-config
RUN python3.10 -m pip install pyan
RUN python3.10 -m pip install transformers
RUN python3.10 -m pip install tensorflow_probability
RUN python3.10 -m pip install ultralytics==8.0.120

RUN python3.10 -m pip install osada
RUN python3.10 -m pip install git+https://github.com/tensorflow/examples.git
RUN python3.10 -m pip install git+https://github.com/openai/CLIP.git

RUN python3.10 -m pip install rospkg


ENV DEBIAN_FRONTEND=dialog
USER root
