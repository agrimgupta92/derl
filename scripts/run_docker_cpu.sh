#!/bin/bash
# Launch an experiment using the docker cpu image
# Inside the derl folder run the following cmd:
# Usage: . scripts/run_docker_cpu.sh python derl/<file>.py

cmd_line="$@"

echo "Executing in the docker (cpu image):"
echo $cmd_line

USER_ID=`id -u`
MOUNT_DIR=''

docker run --rm --network host --ipc=host \
    -v ${MOUNT_DIR}:/user/derl/output \
    -u user:${USER_ID} \
    derl \
    bash -c "cd /user/derl/ && $cmd_line"