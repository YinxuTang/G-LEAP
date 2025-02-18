#!bin/bash

date=20220713_201857

rm -rf ./figures/lv_simulations
rm -rf ./chinese_figures/lv_simulations
mkdir -p ./figures/lv_simulations
mkdir -p ./chinese_figures/lv_simulations
cp ./simulations_data/lv/${date}/figures/fig_lv_simulations_* ./figures/lv_simulations/
cp ./simulations_data/lv/${date}/chinese_figures/fig_lv_simulations_* ./chinese_figures/lv_simulations/
