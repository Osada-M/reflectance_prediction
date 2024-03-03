#!/bin/bash

cd /home/osada/osada_ws/docker_3d
docker image build -t osada:3d .
# docker image build --no-cache -t osada:py_novel .
