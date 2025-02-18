#!/bin/bash

mkdir -p ./simulations_data/lv
# conda activate gleap_env

# Get the begin time
BEGIN=$(date +"%Y%m%d_%H%M%S")
echo "python -u run_lv_simulations.py --date ${BEGIN}"

mkdir -p ./log/lv
setsid stdbuf -i0 -o0 -e0 python -u run_lv_simulations.py --date ${BEGIN} > log/lv/run_lv_simulations_${BEGIN}.log 2>&1 & disown
process_id=$!
echo PID: $process_id
echo $process_id > log/lv/run_lv_simulations_${BEGIN}_pid.txt
tail -f log/lv/run_lv_simulations_${BEGIN}.log
