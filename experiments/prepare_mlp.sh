#!/bin/bash

mkdir -p ./simulations_data/prepare_mlp
# conda activate gleap_env

# Get the begin time
BEGIN=$(date +"%Y%m%d_%H%M%S")
echo "python -u prepare_mlp.py --date ${BEGIN}"

mkdir -p ./simulations_data/prepare_mlp/${BEGIN}/log
setsid stdbuf -i0 -o0 -e0 python -u prepare_mlp.py --date ${BEGIN} > ./simulations_data/prepare_mlp/${BEGIN}/log/prepare_mlp_${BEGIN}.log 2>&1 & disown
process_id=$!
echo PID: $process_id
echo
echo $process_id > ./simulations_data/prepare_mlp/${BEGIN}/log/prepare_mlp_${BEGIN}_pid.txt
tail -f ./simulations_data/prepare_mlp/${BEGIN}/log/prepare_mlp_${BEGIN}.log
