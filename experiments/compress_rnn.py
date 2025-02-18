"""Compresses RNN models via quantization.

Reference: https://pytorch.org/docs/stable/quantization.html

For each RNN model with the filename `${MODEL_NAME}.pkl`, persist the
quantized model into the directory
`./simulations_data/prepare_rnn/${DATE}/compressed_model/${MODEL_NAME}.pkl`.
"""

import copy
from pathlib import Path
import pickle
import time
total_begin_time = time.time()

import torch

from nlp.dataset_utils import load_dataset
from util import format_time, format_now


DATE = "20220724_225255"


# The directory for storing the compressed emoji predictors and models
COMPRESSED_PREDICTOR_DIRECTORY_PATH = "./simulations_data/prepare_rnn/{:s}/compressed_predictor/".format(DATE)
COMPRESSED_MODEL_DIRECTORY_PATH = "./simulations_data/prepare_rnn/{:s}/compressed_model/".format(DATE)
# Create the directory if not exists (mkdir -p)
Path(COMPRESSED_PREDICTOR_DIRECTORY_PATH).mkdir(parents=True, exist_ok=True)
Path(COMPRESSED_MODEL_DIRECTORY_PATH).mkdir(parents=True, exist_ok=True)


# Load the validation dataset
begin_time = time.time()
print("Loading the validation dataset...")
validation_dataset = load_dataset("./dataset/semeval_2018/trial/us_trial")
end_time = time.time()
print("Validation dataset loaded in {:s}".format(format_time(end_time - begin_time)))
print()


def compress_emoji_predictor(path_emoji_predictor):
    # Load the original emoji predictor and the PyTorch model
    with open(path, "rb") as file:
        emoji_predictor = pickle.load(file)
        print("Emoji predictor {:s} loaded.".format(str(path_emoji_predictor)))
    emoji_predictor.load_model()
    # compressed_emoji_predictor = copy.deepcopy(emoji_predictor)
    # compressed_emoji_predictor.load_model()
    # with open(emoji_predictor.model_path, "rb") as file:
    #     trained_model = torch.load()


    # Compress the model via quantization
    compressed_model = torch.quantization.quantize_dynamic(
        emoji_predictor.model, # the original model
        { torch.nn.Linear, torch.nn.RNN }, # a set of layers to dynamically quantize
        dtype=torch.qint8) # the target dtype for quantized weights
    
    
    compressed_emoji_predictor = copy.deepcopy(emoji_predictor)
    compressed_emoji_predictor.model = compressed_model
    compressed_emoji_predictor.validate(validation_dataset)

    # # TODO: Persist the compressed emoji predictor and the compressed PyTorch model
    # compressed_emoji_predictor_path = COMPRESSED_PREDICTOR_DIRECTORY_PATH + "{:s}.pkl".format(
    #     path_emoji_predictor.stem)
    # compressed_model_path = COMPRESSED_MODEL_DIRECTORY_PATH + "{:s}.pt".format(
    #     path_emoji_predictor.stem)
    # compressed_emoji_predictor.unload_model(compressed_model_path)
    # with open(compressed_emoji_predictor_path, "wb") as file:
    #     pickle.dump(compressed_emoji_predictor, file, protocol=5)


path_list = Path("./simulations_data/prepare_rnn/{:s}/predictor".format(DATE)).glob("*.pkl")
for path in path_list:
    compress_emoji_predictor(path)


total_end_time = time.time()
print("{:s}: compress_rnn.py finished in {:s}.".format(format_now(), format_time(total_end_time - total_begin_time)))
