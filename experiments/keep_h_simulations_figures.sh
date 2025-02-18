#!bin/bash

date=20220211_101931

rm -rf ./figures/h_simulations
rm -rf ./chinese_figures/h_simulations
mkdir -p ./figures/h_simulations
mkdir -p ./chinese_figures/h_simulations
cp ./simulations_data/h/${date}/figures/fig_h_simulations_time_averaged_compound_reward.* ./figures/h_simulations/
cp ./simulations_data/h/${date}/chinese_figures/fig_h_simulations_time_averaged_compound_reward.* ./chinese_figures/h_simulations/
