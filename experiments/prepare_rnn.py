"""Prepares the RNN models for emoji prediction.

For each RNN model:
1. Create and train a model on the dataset.
2. Persist the trained model to the file system (`./simulations_data/prepare_rnn/${DATE}/predictor/*.pkl`).
"""


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


# For parsing command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--date', type=str, help='The date (time) when the script start.')
args = parser.parse_args()
DATE = args.date
print("prepare_rnn.py: got argument date: {:s}".format(DATE))


# The directory for storing the trained predictors and models
PREDICTOR_DIRECTORY_PATH = "./simulations_data/prepare_rnn/{:s}/predictor/".format(DATE)
MODEL_DIRECTORY_PATH = "./simulations_data/prepare_rnn/{:s}/model/".format(DATE)
# Create the directory if not exists (mkdir -p)
Path(PREDICTOR_DIRECTORY_PATH).mkdir(parents=True, exist_ok=True)
Path(MODEL_DIRECTORY_PATH).mkdir(parents=True, exist_ok=True)


# The directory for storing the training objects
TRAINING_DIRECTORY_PATH = "./simulations_data/prepare_rnn/{:s}/training/".format(DATE)
# Create the directory if not exists (mkdir -p)
Path(TRAINING_DIRECTORY_PATH).mkdir(parents=True, exist_ok=True)


# Hyperparameters
NUM_EPOCH = 100
# NUM_EPOCH = 1
LEARNING_RATE = 1e-3
MOMENTUM = 0.5
GLOVE_EMBEDDINGS_PATH = "./glove_embeddings/glove.twitter.27B.50d.txt"
print("\nHyperparameters:")
print("  NUM_EPOCH: {:d}".format(NUM_EPOCH))
print("  LEARNING_RATE:", LEARNING_RATE)
print("  MOMENTUM:", MOMENTUM)
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


def train_and_validate_model(
    model,
    training_dataset,
    validation_dataset,
    learning_rate,
    momentum,
    num_epoch,
    notes,
    predictor_dump_filename,
    model_dump_filename):
    begin_time = time.time()
    
    # Train the emoji predictor
    training = Training(
        model,
        training_dataset,
        validation_dataset,
        learning_rate,
        momentum,
        num_epoch,
        notes)
    training.run()
    
    # Dump the predictor
    with open(predictor_dump_filename, "wb") as file:
        # Unload the PyTorch model
        training.neural_emoji_predictor.unload_model(model_dump_filename)

        pickle.dump(training.neural_emoji_predictor, file, protocol=5)
        print("Trained predictor dumped to file \"{:s}\".".format(predictor_dump_filename))
    
    end_time = time.time()
    print("Training finished in {:s}.".format(format_time(end_time - begin_time)))

    return training


training_list = []


# # RNN 1
# # Create the emoji predictor
# rnn = Rnn(glove_vectorizer.vector_length, 20, 32, name="RNN 1")
# print("RNN 1:", rnn)
# rnn_emoji_predictor = RnnEmojiPredictor(vectorizer=copy.deepcopy(glove_vectorizer), model=rnn)
# # Train and validate
# training = train_and_validate_model(
#     rnn_emoji_predictor,
#     copy.deepcopy(training_dataset),
#     copy.deepcopy(validation_dataset),
#     LEARNING_RATE,
#     MOMENTUM,
#     NUM_EPOCH,
#     "RNN 1",
#     PREDICTOR_DIRECTORY_PATH + "rnn_1.pkl",
#     MODEL_DIRECTORY_PATH + "rnn_1.pt")
# training_list.append(training)
# print()


# RNN 2
# Create the emoji predictor
rnn = Rnn(glove_vectorizer.vector_length, 20, 64, name="RNN 2")
print("RNN 2:", rnn)
rnn_emoji_predictor = RnnEmojiPredictor(vectorizer=copy.deepcopy(glove_vectorizer), model=rnn)
# Train and validate
training = train_and_validate_model(
    rnn_emoji_predictor,
    copy.deepcopy(training_dataset),
    copy.deepcopy(validation_dataset),
    LEARNING_RATE,
    MOMENTUM,
    NUM_EPOCH,
    "RNN 2",
    PREDICTOR_DIRECTORY_PATH + "rnn_2.pkl",
    MODEL_DIRECTORY_PATH + "rnn_2.pt")
training_list.append(training)
print()


# # RNN 3
# # Create the emoji predictor
# rnn = Rnn(glove_vectorizer.vector_length, 20, 128, name="RNN 3")
# print("RNN 3:", rnn)
# rnn_emoji_predictor = RnnEmojiPredictor(vectorizer=copy.deepcopy(glove_vectorizer), model=rnn)
# # Train and validate
# training = train_and_validate_model(
#     rnn_emoji_predictor,
#     copy.deepcopy(training_dataset),
#     copy.deepcopy(validation_dataset),
#     LEARNING_RATE,
#     MOMENTUM,
#     NUM_EPOCH,
#     "RNN 3",
#     PREDICTOR_DIRECTORY_PATH + "rnn_3.pkl",
#     MODEL_DIRECTORY_PATH + "rnn_3.pt")
# training_list.append(training)
# print()


# Dump the training list
for training in training_list:
    training.prepare_dump()
filename = TRAINING_DIRECTORY_PATH + "training_list.pkl"
with open(filename, "wb") as file:
    pickle.dump(training_list, file, protocol=5)
    print("training_list dumped to file \"{:s}\".".format(filename))


total_end_time = time.time()
print("{:s}: prepare_rnn.py finished in {:s}.".format(format_now(), format_time(total_end_time - total_begin_time)))
print("DATE: {:s}".format(DATE))
