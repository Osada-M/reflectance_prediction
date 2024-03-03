#!/bin/bash

cd /home/osada/osada_ws/docker_ros
docker image build -t osada:ros .
# docker image build --no-cache -t osada:ros .
