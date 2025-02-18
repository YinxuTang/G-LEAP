import argparse
import copy
from pathlib import Path
import pickle
import time
total_begin_time = time.time()

from nlp.dataset_utils import load_dataset
from nlp.vectorizer import GloveVectorizer
from nlp.neural import Rnn, RnnEmojiPredictor, Training
from util import format_time, format_now


# Hyperparameters
DATE = "20220724_225255"
print("\nHyperparameters:")
print("  DATE: \"{:s}\"".format(DATE))
print()


# The directory for storing the compressed emoji predictors
COMPRESSED_PREDICTOR_DIRECTORY_PATH = "./simulations_data/prepare_rnn/{:s}/compressed_predictor/".format(DATE)


# Load the validation dataset
begin_time = time.time()
print("Loading the validation dataset...")
validation_dataset = load_dataset("./dataset/semeval_2018/trial/us_trial")
end_time = time.time()
print("Validation dataset loaded in {:s}".format(format_time(end_time - begin_time)))
print()


path_list = Path(COMPRESSED_PREDICTOR_DIRECTORY_PATH).glob("*.pkl")
for path in path_list:
    print(path)
    with open(path, "rb") as file:
        emoji_predictor = pickle.load(file)
    emoji_predictor.load_model()
    emoji_predictor.validate(validation_dataset)


total_end_time = time.time()
print("{:s}: evaluate_compressed_rnn.py finished in {:s}.".format(format_now(), format_time(total_end_time - total_begin_time)))
