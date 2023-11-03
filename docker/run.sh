#!/bin/bash
CONTAINER_NAME="predrnn"
DOCKER_ENV=""
PYTORCH_VERSION="1.9.0"
CUDA_VERSION="11.1"
CUDNN_VERSION="8"
IMAGE_FLAVOR="devel"
OPENCV_VERSION="4.7.0.68"
USER_ID=$(id -u)
CMDNAME=$(basename $0)
PYTORCH_IMAGE_NAME="predrnn:${PYTORCH_VERSION}-cuda${CUDA_VERSION}-cudnn${CUDNN_VERSION}-${IMAGE_FLAVOR}-opencv${OPENCV_VERSION}"
XSOCK="/tmp/.X11-unix"
XAUTH="/tmp/.docker.xauth"
HOST_WS=$(dirname $(dirname $(readlink -f $0)))/shared_dir
DOCKER_VOLUME="-v ${XSOCK}:${XSOCK}:rw"
DOCKER_VOLUME="${DOCKER_VOLUME} -v ${XAUTH}:${XAUTH}:rw"
DOCKER_VOLUME="${DOCKER_VOLUME} -v ${HOST_WS}:/home/predrnn/shared_dir:rw"
DOCKER_ENV="-e XAUTHORITY=${XAUTH}"
DOCKER_ENV="${DOCKER_ENV} -e DISPLAY=$DISPLAY"
DOCKER_ENV="${DOCKER_ENV} -e QT_X11_NO_MITSHM=1"
DOCKER_ENV="${DOCKER_ENV} -e USER_ID=${USER_ID}"
DOCKER_ENV="${DOCKER_ENV} -e HOME=/home/predrnn"
DOCKER_IMAGE="${PYTORCH_IMAGE_NAME}"
CONTAINER_CMD="/bin/bash"
# CONTAINER_CMD="/home/predrnn/.bashrc"
DOCKER_NET="host"
clear
printf "\033[01;31m\n"
printf " ________          ________                      ______ \n";
printf " ___  __ \_____  _____  __/______ __________________  /_ \n";
printf " __  /_/ /__  / / /__  /   _  __ \__  ___/_  ___/__  __ \ \n";
printf " _  ____/ _  /_/ / _  /    / /_/ /_  /    / /__  _  / / / \n";
printf " /_/      _\__, /  /_/     \____/ /_/     \___/  /_/ /_/ \n";
printf "          /____/ ";
printf "\n"
printf "\n"
printf "\033[00m\n"
touch ${XAUTH}
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f ${XAUTH} nmerge -
docker run \
  --rm -it \
  --gpus all \
  --privileged \
  --name ${CONTAINER_NAME} \
  --net ${DOCKER_NET} \
  --shm-size 10gb \
  --user ${USER_ID} \
  ${DOCKER_ENV} \
  ${DOCKER_VOLUME} \
  ${DOCKER_IMAGE} \
  ${CONTAINER_CMD}
