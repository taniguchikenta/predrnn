#!/bin/bash
PYTORCH_VERSION="1.9.0"
CUDA_VERSION="11.1"
CUDNN_VERSION="8"
IMAGE_FLAVOR="devel"
OPENCV_VERSION="4.7.0.68"
BUILD_DIR=$(dirname $(readlink -f $0))/src
USER_ID=$(id -u)
PYTORCH_IMAGE_NAME="predrnn:${PYTORCH_VERSION}-cuda${CUDA_VERSION}-cudnn${CUDNN_VERSION}-${IMAGE_FLAVOR}-opencv${OPENCV_VERSION}"
docker build \
  -t ${PYTORCH_IMAGE_NAME} \
  -f ${BUILD_DIR}/Dockerfile \
  --build-arg UID=$(id -u) \
  --build-arg GID=$(id -g) \
  --build-arg HOSTNAME=$(hostname) \
  ${BUILD_DIR}
