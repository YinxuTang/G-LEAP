"""Prepares the LSTM models for emoji prediction.

For each LSTM model:
1. Create and train a model on the dataset.
2. Persist the trained model to the file system (`./simulations_data/prepare_lstm/${DATE}/predictor/*.pkl`).
"""


import argparse
import copy
from pathlib import Path
import pickle
import time
total_begin_time = time.time()

from nlp.dataset_utils import load_dataset
from nlp.vectorizer import GloveVectorizer
from nlp.neural import Lstm, LstmEmojiPredictor, Training
from util import format_time, format_now


# For parsing command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--date', type=str, help='The date (time) when the script start.')
args = parser.parse_args()
DATE = args.date
print("prepare_lstm.py: got argument date: {:s}".format(DATE))


# The directory for storing the trained models
MODEL_DIRECTORY_PATH = "./simulations_data/prepare_lstm/{:s}/predictor/".format(DATE)
# Create the directory if not exists (mkdir -p)
Path(MODEL_DIRECTORY_PATH).mkdir(parents=True, exist_ok=True)


# The directory for storing the training objects
TRAINING_DIRECTORY_PATH = "./simulations_data/prepare_lstm/{:s}/training/".format(DATE)
# Create the directory if not exists (mkdir -p)
Path(TRAINING_DIRECTORY_PATH).mkdir(parents=True, exist_ok=True)


# Hyperparameters
NUM_EPOCH = 100
LEARNING_RATE_LIST = [ 1e-3, 1e-3, 1e-3 ]
MOMENTUM_LIST = [ 0.5, 0.5, 0.5 ]
GLOVE_EMBEDDINGS_PATH = "./glove_embeddings/glove.twitter.27B.50d.txt"
print("\nHyperparameters:")
print("  NUM_EPOCH: {:d}".format(NUM_EPOCH))
print("  LEARNING_RATE_LIST:", LEARNING_RATE_LIST)
print("  MOMENTUM_LIST:", MOMENTUM_LIST)
print("  GLOVE_EMBEDDINGS_PATH: \"{:s}\"".format(GLOVE_EMBEDDINGS_PATH))
print()


# Load the datasets
begin_time = time.time()
print("Loading the datasets...")
training_dataset = load_dataset("./dataset/semeval_2018/train/us_train")
validation_dataset = load_dataset("./dataset/semeval_2018/trial/us_trial")
end_time = time.time()
print("Datasets loaded in {:s}".format(format_time(end_time - begin_time)))
print()


# The vectorizer is only created once, and the copies are used
glove_vectorizer = GloveVectorizer(glove_embeddings_path=GLOVE_EMBEDDINGS_PATH)


def train_and_validate_model(model, training_dataset, validation_dataset, learning_rate, momentum, num_epoch, notes, predictor_dump_filename):
    begin_time = time.time()
    
    # Train the emoji predictor
    training = Training(model, training_dataset, validation_dataset, learning_rate, momentum, num_epoch, notes)
    training.run()
    
    # Dump the predictor
    with open(predictor_dump_filename, "wb") as file:
        pickle.dump(training.neural_emoji_predictor, file, protocol=5)
        print("Trained predictor dumped to file \"{:s}\".".format(predictor_dump_filename))
    
    end_time = time.time()
    print("Training finished in {:s}.".format(format_time(end_time - begin_time)))

    return training


training_list = []
predictor_index = 0


# LSTM 1
# Create the emoji predictor
lstm = Lstm(glove_vectorizer.vector_length, 20, 32, num_layer=1, bidirectional=False, name="LSTM-1")
print("LSTM-1:", lstm)
lstm_emoji_predictor = LstmEmojiPredictor(vectorizer=copy.deepcopy(glove_vectorizer), model=lstm)
# Train and validate
training = train_and_validate_model(
    lstm_emoji_predictor,
    copy.deepcopy(training_dataset),
    copy.deepcopy(validation_dataset),
    LEARNING_RATE_LIST[predictor_index],
    MOMENTUM_LIST[predictor_index],
    NUM_EPOCH,
    "LSTM 1",
    MODEL_DIRECTORY_PATH + "lstm_1.pkl")
training_list.append(training)
predictor_index += 1
print()


# LSTM 2
# Create the emoji predictor
lstm = Lstm(glove_vectorizer.vector_length, 20, 32, num_layer=2, bidirectional=False, name="LSTM-2")
print("LSTM-2:", lstm)
lstm_emoji_predictor = LstmEmojiPredictor(vectorizer=copy.deepcopy(glove_vectorizer), model=lstm)
# Train and validate
training = train_and_validate_model(
    lstm_emoji_predictor,
    copy.deepcopy(training_dataset),
    copy.deepcopy(validation_dataset),
    LEARNING_RATE_LIST[predictor_index],
    MOMENTUM_LIST[predictor_index],
    NUM_EPOCH,
    "LSTM 2",
    MODEL_DIRECTORY_PATH + "lstm_2.pkl")
training_list.append(training)
predictor_index += 1
print()


# LSTM 3
# Create the emoji predictor
lstm = Lstm(glove_vectorizer.vector_length, 20, 32, num_layer=2, bidirectional=True, name="Bi-LSTM")
print("Bi-LSTM:", lstm)
lstm_emoji_predictor = LstmEmojiPredictor(vectorizer=copy.deepcopy(glove_vectorizer), model=lstm)
# Train and validate
training = train_and_validate_model(
    lstm_emoji_predictor,
    copy.deepcopy(training_dataset),
    copy.deepcopy(validation_dataset),
    LEARNING_RATE_LIST[predictor_index],
    MOMENTUM_LIST[predictor_index],
    NUM_EPOCH,
    "Bi-LSTM",
    MODEL_DIRECTORY_PATH + "bi_lstm.pkl")
training_list.append(training)
predictor_index += 1
print()


# Dump the training list
for training in training_list:
    training.prepare_dump()
filename = TRAINING_DIRECTORY_PATH + "training_list.pkl"
with open(filename, "wb") as file:
    pickle.dump(training_list, file, protocol=5)
    print("training_list dumped to file \"{:s}\".".format(filename))


total_end_time = time.time()
print("{:s}: prepare_lstm.py finished in {:s}.".format(format_now(), format_time(total_end_time - total_begin_time)))
print("DATE: {:s}".format(DATE))
