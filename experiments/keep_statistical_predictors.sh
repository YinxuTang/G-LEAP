#!bin/bash

DATE=20220124_092935

rm -r ./trained_emoji_predictors/statistical
mkdir -p ./trained_emoji_predictors/statistical
cp ./simulations_data/prepare_statistical/${DATE}/predictor/*.pkl ./trained_emoji_predictors/statistical/
