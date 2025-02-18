#!bin/bash

date=20220123_185625

rm -rf ./figures/training/mlp
mkdir -p ./figures/training/mlp
cp ./simulations_data/prepare_mlp/${date}/figures/* ./figures/training/mlp/
