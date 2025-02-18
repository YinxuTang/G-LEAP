import argparse
import copy
import pickle
import time

import numpy as np

import simulation_config
from mab import EmojiPredictionEnv, Simulation, GleapAgent, GleapEpsilonGreedyAgent, UcbAgent, UniformlyRandomAgent
from util import format_time
from nlp.dataset_utils import load_dataset


total_begin_time = time.time()


# Hyperparameters
MOCK_PREDICTION = True
BETA = simulation_config.BETA
NUM_ARM = simulation_config.NUM_ARM
NUM_SELECTION = simulation_config.NUM_SELECTION
NUM_ROUND = simulation_config.NUM_ROUND
NUM_REPETITION = simulation_config.NUM_REPETITION
LATENCY_MEAN_LIST = simulation_config.LATENCY_MEAN_LIST
E_ARRAY = simulation_config.E_ARRAY
V = simulation_config.V
# V_ARRAY = np.array([0.1, 1.0, 10.0, 100.0, 1000.0])
# V_ARRAY = np.array([ 1, 10, 100, 1000, 10000, 100000 ])
# V_ARRAY = np.arange(100, 1001, 100)
# V_ARRAY[0] = 1.0
# B = simulation_config.B
# B_ARRAY = np.arange(10, 25, 1)
# B_ARRAY = np.arange(12.0, 15.1, 0.5)
B_ARRAY = np.arange(11.0, 16.1, 0.1)
H = simulation_config.H
GAMMA = simulation_config.GAMMA


# Print the setting of the hyperparameters
print("---------Hyperparameters----------")
print("    MOCK_PREDICTION:", MOCK_PREDICTION)
print("    BETA: {:f}".format(BETA))
print("    NUM_ARM: {:d}".format(NUM_ARM))
print("    NUM_SELECTION: {:d}".format(NUM_SELECTION))
print("    NUM_ROUND: {:d}".format(NUM_ROUND))
print("    NUM_REPETITION: {:d}".format(NUM_REPETITION))
print("    LATENCY_MEAN_LIST:", LATENCY_MEAN_LIST)
print("    E_ARRAY:", E_ARRAY)
print("    V:", V)
print("    B_ARRAY:", B_ARRAY)
print("    H: {:d}".format(H))
print("    GAMMA: {:f}".format(GAMMA))
print("------End of Hyperparameters------")


# Load all the inference models
begin_time = time.time()
if not MOCK_PREDICTION:
    with open("./emoji_predictor_list.pkl", "rb") as file:
        predictor_list = pickle.load(file)
else:
    predictor_list = []
# Load arm_predictor_index_mapping_list
with open("./arm_predictor_index_mapping_list.pkl", "rb") as f:
    arm_predictor_index_mapping_list = pickle.load(f)
# Load accuracy_list
with open("./accuracy_list.pkl", "rb") as f:
    accuracy_list = pickle.load(f)
# Load history information
with open("./history_information.pkl", "rb") as f:
    history_information = pickle.load(f)
# Load the user input corpus (the validation dataset TODO: use the test dataset)
user_input_corpus = load_dataset("./dataset/semeval_2018/trial/us_trial")
end_time = time.time()
print("{:d} inference models, arm_predictor_index_mapping_list, history_information, and user_input_corpus loaded in {:s}.".format(len(predictor_list), format_time(end_time - begin_time)))


print("accuracy_list:", accuracy_list)
print("The minimum feasible budget b = {:.2f}".format(sum(sorted(E_ARRAY)[:NUM_SELECTION])))
print("The energy consumption of uniformly random: {:.2f}".format(np.mean(E_ARRAY) * NUM_SELECTION))

# # G-LEAP agents with different values of budget
# for b in B_ARRAY:
#     print("b = {:.2f}".format(b))
#     # Build the environment
#     env = EmojiPredictionEnv(name="EmojiPredictionEnv", beta=BETA, num_arm=NUM_ARM, num_selection=NUM_SELECTION, predictor_list=predictor_list, accuracy_list=accuracy_list, arm_predictor_index_mapping_list=arm_predictor_index_mapping_list, user_input_corpus=user_input_corpus, latency_mean_list=LATENCY_MEAN_LIST, e_array=E_ARRAY, b=b, mock_prediction=MOCK_PREDICTION)
#     print()


total_end_time = time.time()
print("Evaluation for b settings finished in {:s}.".format(format_time(total_end_time - total_begin_time)))
