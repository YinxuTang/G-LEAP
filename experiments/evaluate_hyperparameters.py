import pickle
import time

import numpy as np

from mab import EmojiPredictionEnv, Simulation, GleapAgent, GleapEpsilonGreedyAgent, UcbAgent, UniformlyRandomAgent, HEpsilonGreedyAgent
import simulation_config
from util import format_time
from util import arg_top_k


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
B = simulation_config.B
GAMMA = simulation_config.GAMMA
EPSILON = simulation_config.EPSILON


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
print("    V: {:f}".format(V))
print("    B: {:f}".format(B))
print("    GAMMA: {:f}".format(GAMMA))
print("    EPSILON: {:f}".format(EPSILON))
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
    print("arm_predictor_index_mapping_list:", arm_predictor_index_mapping_list)
# Load accuracy_list
with open("./accuracy_list.pkl", "rb") as f:
    accuracy_list = pickle.load(f)
# Load history information
with open("./history_information.pkl", "rb") as f:
    history_information = pickle.load(f)
# Load the user input corpus
with open("./dataset/test_dataset.pkl", "rb") as f:
    user_input_corpus = pickle.load(f)
end_time = time.time()
print("{:d} inference models, arm_predictor_index_mapping_list, history_information and user_input_corpus loaded in {:s}.".format(len(predictor_list), format_time(end_time-begin_time)))


# Print the mean list of the normalized reward
def display_normalized_reward_mean_list():
    normalized_reward_mean_list = []

    # Calculate the bounds of the latency
    latency_std = 200
    latency_lower_bound = 2
    latency_fluctuation = 150
    d_min = max(min(LATENCY_MEAN_LIST) - latency_fluctuation, latency_lower_bound)
    d_max = max(LATENCY_MEAN_LIST) + latency_fluctuation

    # For the normalization of rewards (raw observations)
    loc = BETA * d_max
    scale = 1 / (1 + BETA * (d_max - d_min))

    for arm_index in range(NUM_ARM):
        # Get the predictor index
        predictor_index = arm_predictor_index_mapping_list[arm_index]

        # Get the accuracy
        accuracy = accuracy_list[predictor_index]

        # Get the latency mean
        latency_mean = LATENCY_MEAN_LIST[predictor_index]

        normalized_reward_mean = (accuracy + BETA * (d_max - latency_mean)) / (1 + BETA * (d_max - d_min))
        normalized_reward_mean_list.append(normalized_reward_mean)

    print("Settings of each arm:")
    for index, normalized_reward_mean in enumerate(normalized_reward_mean_list):
        predictor_index = arm_predictor_index_mapping_list[index]
        print("  Arm {:d}: reward mean = {:.2f}, energy consumption = {:.2f}".format(index, normalized_reward_mean, E_ARRAY[predictor_index]))
    
    return normalized_reward_mean_list
print()
normalized_reward_mean_list = display_normalized_reward_mean_list()
print()


# Build the environment to calculate the optimal compound reward
env = EmojiPredictionEnv(name="EmojiPredictionEnv", beta=BETA, num_arm=NUM_ARM, num_selection=NUM_SELECTION, predictor_list=predictor_list, accuracy_list=accuracy_list, arm_predictor_index_mapping_list=arm_predictor_index_mapping_list, user_input_corpus=user_input_corpus, latency_mean_list=LATENCY_MEAN_LIST, e_array=E_ARRAY, b=B, mock_prediction=MOCK_PREDICTION)
print("Expected compound reward of optimal: {:.2f}".format(env.optimal_r))


# For the random
random_expected_compound_reward = NUM_SELECTION * np.mean(normalized_reward_mean_list)
print("Expected compound reward of UniformlyRandomAgent: {:.2f}".format(random_expected_compound_reward))


# For the online learning (without online control) algorithms
optimal_indices_for_learning_without_control = arg_top_k(normalized_reward_mean_list, NUM_SELECTION)
learning_without_control_expected_compound_reward = 0.0
for index in optimal_indices_for_learning_without_control:
    learning_without_control_expected_compound_reward += normalized_reward_mean_list[index]
print("Expected compound reward of optimal learning without control: {:.2f}".format(learning_without_control_expected_compound_reward))
print("  Optimal arm indices for learning without control:", optimal_indices_for_learning_without_control)


# Check for the feasibility
# We say that a setting of the hyperparameters is feasible if and only if some specified conditions are satisfied.
# These conditions are checked in the following lines of code
feasible = True
feasible &= random_expected_compound_reward < env.optimal_r
feasible &= learning_without_control_expected_compound_reward > env.optimal_r
print("\nHyperparameter feasible:", feasible)
