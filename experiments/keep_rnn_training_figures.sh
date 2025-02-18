#!bin/bash

DATE=20220124_110822

rm -rf ./figures/training/rnn
mkdir -p ./figures/training/rnn
cp ./simulations_data/prepare_rnn/${DATE}/figures/* ./figures/training/rnn/
