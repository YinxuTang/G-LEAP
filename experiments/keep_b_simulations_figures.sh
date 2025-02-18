#!bin/bash

date=20220222_213836

rm -rf ./figures/b_simulations
rm -rf ./chinese_figures/b_simulations
mkdir -p ./figures/b_simulations
mkdir -p ./chinese_figures/b_simulations
cp ./simulations_data/b/${date}/figures/fig_b_simulations_* ./figures/b_simulations/
cp ./simulations_data/b/${date}/chinese_figures/fig_b_simulations_* ./chinese_figures/b_simulations/
