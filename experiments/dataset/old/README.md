# Dataset Related Files

## Twitter Emoji Prediction Dataset

- `mapping.csv`
- `output_format.csv`
- `test.csv`
- `train.csv`

These four files are copied from [Twitter Emoji Prediction Dataset](https://www.kaggle.com/hariharasudhanas/twitter-emoji-prediction). As the original test dataset lacks true labels, we need to treat the original train dataset as the whole dataset and split it into the train dataset and the test dataset for our experiments. The split datasets are `train_dataset.pkl` and `test_dataset.pkl`. Each resulting dataset represents a list, each item of whom is a tuple: `(text, label)`, where `text` is of type `str` and `label` is of type `int`.

## Labels Output by Pretrained Cardiffnlp Transformer Models

- `cardiffnlp_transformer_models_mapping.txt`: The raw mapping file.
- `cardiffnlp_to_dataset_label_index_mapping_list.pkl`: This file is generated using the script `compute_cardiffnlp_models_label_mapping.py`.

Source: https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/emoji/mapping.txt
