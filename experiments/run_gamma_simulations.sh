#!/bin/bash

mkdir -p ./simulations_data/gamma
# conda activate gleap_env

# Get the begin time
BEGIN=$(date +"%Y%m%d_%H%M%S")
echo "python -u run_gamma_simulations.py --date ${BEGIN}"

mkdir -p ./log/gamma
setsid stdbuf -i0 -o0 -e0 python -u run_gamma_simulations.py --date ${BEGIN} > log/gamma/run_gamma_simulations_${BEGIN}.log 2>&1 & disown
process_id=$!
echo PID: $process_id
echo $process_id > log/gamma/run_gamma_simulations_${BEGIN}_pid.txt
tail -f log/gamma/run_gamma_simulations_${BEGIN}.log
