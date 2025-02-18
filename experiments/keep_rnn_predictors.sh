#!bin/bash

DATE=20220124_110822

rm -rf ./trained_emoji_predictors/neural/rnn
mkdir -p ./trained_emoji_predictors/neural/rnn
cp ./simulations_data/prepare_rnn/${DATE}/predictor/*.pkl ./trained_emoji_predictors/neural/rnn
