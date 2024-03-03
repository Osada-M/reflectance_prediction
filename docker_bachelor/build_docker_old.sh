#!/bin/bash

if [ $# -ne 1 ]; then
    docker images
    echo;
    echo "タグ名の入力が必要" 1>&2
    exit 1
fi


cd /home/osada/osada_ws/docker
docker image build -t osada:$1 .
# docker image build --no-cache -t osada:$1 .
