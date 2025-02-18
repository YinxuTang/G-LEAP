#!bin/bash

DATE=20220123_185625

rm -rf ./trained_emoji_predictors/neural/mlp
mkdir -p ./trained_emoji_predictors/neural/mlp
cp ./simulations_data/prepare_mlp/${DATE}/predictor/*.pkl ./trained_emoji_predictors/neural/mlp
