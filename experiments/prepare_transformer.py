"""Prepares the transformer-based emoji predictors.

1. Download and save the transformer models from the Huggingface Python `transformers` library to `./transformer_models/`.
2. Construct the corresponding `nlp.neural.TransformerEmojiPredictor` objects and save them to the final directory `./trained_emoji_predictors/neural/transformer`.
"""


from pathlib import Path
import pickle
import time
total_begin_time = time.time()
import datetime

from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

from util import format_time, format_now
from nlp.neural import TransformerEmojiPredictor


# The directory for storing the models from the Huggingface Python transformers library
MODEL_DIRECTORY_PATH = "./transformer_models/"
# Create the directory if not exists (mkdir -p)
Path(MODEL_DIRECTORY_PATH).mkdir(parents=True, exist_ok=True)


# The directory for storing the trained predictors
PREDICTOR_DIRECTORY_PATH = "./trained_emoji_predictors/neural/transformer/"
# Create the directory if not exists (mkdir -p)
Path(PREDICTOR_DIRECTORY_PATH).mkdir(parents=True, exist_ok=True)


def download_and_save_model(model_name):
    # Download the model
    print("Downloading the model: {:s}...".format(model_name))
    model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Downloading completed.")

    # Save the model
    # save_path = "./transformer_models/{:s}".format(model_name)
    save_path = MODEL_DIRECTORY_PATH + model_name
    print("Saving the model into: {:s}...".format(save_path))
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)
    print("Saving completed.")


# BERT ("cardiffnlp/bertweet-base-emoji")
model_name = "cardiffnlp/bertweet-base-emoji"
save_path = MODEL_DIRECTORY_PATH + model_name
download_and_save_model(model_name)
tokenizer = AutoTokenizer.from_pretrained(save_path)
model = AutoModelForSequenceClassification.from_pretrained(save_path).eval()
emoji_predictor = TransformerEmojiPredictor(tokenizer, model, name="BERT")
filename = PREDICTOR_DIRECTORY_PATH + "bert.pkl"
with open(filename, "wb") as file:
    pickle.dump(emoji_predictor, file, protocol=5)
    print("The emoji predictor \"{:s}\" has been dumped to the file \"{:s}\".".format(emoji_predictor.name, filename))


# RoBERTa ("cardiffnlp/twitter-roberta-base-emoji")
model_name = "cardiffnlp/twitter-roberta-base-emoji"
save_path = MODEL_DIRECTORY_PATH + model_name
download_and_save_model(model_name)
tokenizer = AutoTokenizer.from_pretrained(save_path)
model = AutoModelForSequenceClassification.from_pretrained(save_path).eval()
emoji_predictor = TransformerEmojiPredictor(tokenizer, model, name="RoBERTa")
filename = PREDICTOR_DIRECTORY_PATH + "roberta.pkl"
with open(filename, "wb") as file:
    pickle.dump(emoji_predictor, file, protocol=5)
    print("The emoji predictor \"{:s}\" has been dumped to the file \"{:s}\".".format(emoji_predictor.name, filename))


total_end_time = time.time()
print("{:s}: prepare_transformer.py finished in {:s}.".format(format_now(), format_time(total_end_time - total_begin_time)))
