#!/bin/bash

USER_ID=${LOCAL_UID:-9001}
GROUP_ID=${LOCAL_GID:-9001}

echo "Starting with UID : $USER_ID, GID: $GROUP_ID"
useradd -u $USER_ID -o -m user
groupmod -o -g $GROUP_ID user
export HOME=/home/user


echo "export TF_CPP_MIN_LOG_LEVEL=3" >> /home/user/.bashrc

. /home/user/.bashrc

exec /usr/sbin/gosu user "$@"


apt -y install wget

wget https://www.python.org/ftp/python/3.10.6/Python-3.10.6.tar.xz
tar xJf Python-3.10.6.tar.xz
cd Python-3.10.6
./configure
make
make install
