#!bin/bash

date=20220713_162932

rm -rf ./figures/v_simulations
rm -rf ./chinese_figures/v_simulations
mkdir -p ./figures/v_simulations
mkdir -p ./chinese_figures/v_simulations
cp ./simulations_data/v/${date}/figures/fig_v_simulations* ./figures/v_simulations/
cp ./simulations_data/v/${date}/chinese_figures/fig_v_simulations* ./chinese_figures/v_simulations/
