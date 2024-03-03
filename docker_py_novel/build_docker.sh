#!/bin/bash

cd /home/osada/osada_ws/docker_py_novel
docker image build -t osada:py_novel .
# docker image build --no-cache -t osada:py_novel .
