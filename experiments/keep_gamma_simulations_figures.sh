#!bin/bash

date=20220127_223730

rm -rf ./figures/gamma_simulations
rm -rf ./chinese_figures/gamma_simulations
mkdir -p ./figures/gamma_simulations
mkdir -p ./chinese_figures/gamma_simulations
cp ./simulations_data/gamma/${date}/figures/fig_gamma_simulations_time_averaged_compound_reward.* ./figures/gamma_simulations/
cp ./simulations_data/gamma/${date}/chinese_figures/fig_gamma_simulations_time_averaged_compound_reward.* ./chinese_figures/gamma_simulations/
