#!bin/bash

date=20220127_151410

rm -rf ./figures/ht_simulations
rm -rf ./chinese_figures/ht_simulations
mkdir -p ./figures/ht_simulations
mkdir -p ./chinese_figures/ht_simulations
cp ./simulations_data/ht/${date}/figures/fig_ht_simulations_converged_time_averaged_reward.* ./figures/ht_simulations/
cp ./simulations_data/ht/${date}/chinese_figures/fig_ht_simulations_converged_time_averaged_reward.* ./chinese_figures/ht_simulations/
