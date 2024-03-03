#!/bin/bash

USER_ID=${LOCAL_UID:-9001}
GROUP_ID=${LOCAL_GID:-9001}

echo "Starting with UID : $USER_ID, GID: $GROUP_ID"
useradd -u $USER_ID -o -m user
groupmod -o -g $GROUP_ID user
export HOME=/home/user


echo "export TF_CPP_MIN_LOG_LEVEL=3" >> /home/user/.bashrc
echo "alias python='python3.10'" >> /home/user/.bashrc
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc

. /home/user/.bashrc

exec /usr/sbin/gosu user "$@"
