#!/bin/bash

workspace="/home/osada/osada_ws"

if [ $# -ne 1 ]; then
    gpu="all"
else
    gpu=$1
fi

xhost +

docker container run \
--rm --gpus "device=$gpu" -it \
-v $workspace/:/workspace/osada_ws \
-v /home/osada/:/workspace/home_osada \
-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
-e DISPLAY=$DISPLAY \
-e LOCAL_UID=$(id -u $USER) -e LOCAL_GID=$(id -g $USER) \
osada:3d bash
