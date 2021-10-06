#!/bin/bash
set -x

PARENT=ubuntu:16.04
TORCH=cpuonly
TAG=derl
USER_ID=`id -u`

docker build -f docker/Dockerfile \
  --build-arg PARENT_IMAGE=${PARENT} \
  --build-arg TORCH=${TORCH} \
  --build-arg USER_ID=${USER_ID} \
  -t ${TAG} .