#!/bin/bash

mkdir -p ./simulations_data/evaluate_predictors
# conda activate gleap_env

# Get the begin time
BEGIN=$(date +"%Y%m%d_%H%M%S")
echo "python -u evaluate_predictors.py --date ${BEGIN}"

mkdir -p ./simulations_data/evaluate_predictors/${BEGIN}/log
setsid stdbuf -i0 -o0 -e0 python -u evaluate_predictors.py --date ${BEGIN} > ./simulations_data/evaluate_predictors/${BEGIN}/log/evaluate_predictors_${BEGIN}.log 2>&1 & disown
process_id=$!
echo PID: $process_id
echo
echo $process_id > ./simulations_data/evaluate_predictors/${BEGIN}/log/evaluate_predictors_${BEGIN}_pid.txt
tail -f ./simulations_data/evaluate_predictors/${BEGIN}/log/evaluate_predictors_${BEGIN}.log
