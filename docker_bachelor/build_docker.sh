#!/bin/bash

cd /home/osada/osada_ws/docker
docker image build -t osada:bachelor .
# docker image build --no-cache -t osada:bachelor .
