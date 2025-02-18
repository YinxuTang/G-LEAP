import pickle
import time

import numpy as np

from mab import EmojiPredictionEnv, GleapEpsilonGreedyAgent, Simulation
from util import format_time
import simulation_config
from nlp.dataset_utils import load_dataset


# Hyperparameters
MOCK_PREDICTION = True
# MOCK_PREDICTION = False


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
with open("accuracy_list.pkl", "rb") as f:
    accuracy_list = pickle.load(f)
# Load the user input corpus (the validation dataset TODO: use the test dataset)
user_input_corpus = load_dataset("./dataset/semeval_2018/trial/us_trial")
end_time = time.time()
print("{:d} emoji predictors, arm_predictor_index_mapping_list and user_input_corpus loaded in {:s}.".format(len(predictor_list), format_time(end_time - begin_time)))


# Print the setting of the hyperparameters
print("---------Hyperparameters----------")
print("    MOCK_PREDICTION:", MOCK_PREDICTION)
print("    BETA: {:f}".format(simulation_config.BETA))
print("    NUM_ARM: {:d}".format(simulation_config.NUM_ARM))
print("    NUM_SELECTION: {:d}".format(simulation_config.NUM_SELECTION))
print("    NUM_ROUND: {:d}".format(simulation_config.NUM_ROUND))
print("    NUM_REPETITION: {:d}".format(simulation_config.NUM_REPETITION))
print("    LATENCY_MEAN_LIST: ", simulation_config.LATENCY_MEAN_LIST)
print("    E_ARRAY:", simulation_config.E_ARRAY)
print("    V: {:f}".format(simulation_config.V))
print("    B: {:f}".format(simulation_config.B))
print("    H_MAX: {:d}".format(simulation_config.H_MAX))
print("------End of Hyperparameters------")


def display_arm_settings():
    NUM_ARM = simulation_config.NUM_ARM
    E_ARRAY = simulation_config.E_ARRAY

    print("Settings of each arm:")
    for arm_index in range(NUM_ARM):
        predictor_index = arm_predictor_index_mapping_list[arm_index]
        print("  Arm #{:d}: energy consumption = {:.2f}".format(arm_index, E_ARRAY[predictor_index]))
print()
display_arm_settings()
print()


# Build the environment
env = EmojiPredictionEnv(
    name="EmojiPredictionEnv",
    beta=simulation_config.BETA,
    num_arm=simulation_config.NUM_ARM,
    num_selection=simulation_config.NUM_SELECTION,
    predictor_list=predictor_list,
    accuracy_list=accuracy_list,
    arm_predictor_index_mapping_list=arm_predictor_index_mapping_list,
    latency_mean_list=simulation_config.LATENCY_MEAN_LIST,
    latency_std=simulation_config.LATENCY_STD,
    latency_min=simulation_config.LATENCY_MIN,
    latency_max=simulation_config.LATENCY_MAX,
    user_input_corpus=user_input_corpus,
    e_array=simulation_config.E_ARRAY,
    b=simulation_config.B,
    mock_prediction=MOCK_PREDICTION)


begin_time = time.time()
history_information = env.generate_history_information(simulation_config.H_MAX, simulation_config.NUM_SELECTION, log=True, num_process=50)
end_time = time.time()
print("History information generated in {:s}.".format(format_time(end_time - begin_time)))
print("history_information['c'].shape:", history_information['c'].shape)
print("history_information['d'].shape:", history_information['d'].shape)
print("history_information['click'].shape:", history_information['click'].shape)


# Dump the history information
filename = "./history_information.pkl"
with open(filename, "wb") as f:
    pickle.dump(history_information, f, protocol=5)
    print("History information has been dumped to file \"{:s}\".".format(filename))
