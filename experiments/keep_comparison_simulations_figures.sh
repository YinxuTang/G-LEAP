#!bin/bash

date=20220713_161313

rm -rf ./figures/comparison_simulations
rm -rf ./chinese_figures/comparison_simulations
mkdir -p ./figures/comparison_simulations
mkdir -p ./chinese_figures/comparison_simulations
cp ./simulations_data/comparison/${date}/figures/fig_comparison_simulations_* ./figures/comparison_simulations
cp ./simulations_data/comparison/${date}/chinese_figures/fig_comparison_simulations_* ./chinese_figures/comparison_simulations
