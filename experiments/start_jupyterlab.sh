#!/bin/bash

BEGIN=$(date +"%Y%m%d_%H%M%S")
mkdir -p ./log/jupyter/${BEGIN}
setsid stdbuf -i0 -o0 -e0 jupyter lab --config jupyterlab_config.py --ip 0.0.0.0 --port 41562 > ./log/jupyter/${BEGIN}/jupyter.log 2>&1 &
PROCESS_ID=$!
echo PID: ${PROCESS_ID}
echo ${PROCESS_ID} > ./log/jupyter/${BEGIN}/jupyter_pid.txt
tail -f ./log/jupyter/${BEGIN}/jupyter.log
