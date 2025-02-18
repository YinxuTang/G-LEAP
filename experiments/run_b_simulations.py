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


# For parsing command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--date', type=str, help='The date (time) when the script start.')
args = parser.parse_args()
date = args.date
print("Running simulations for start date: {:s}".format(date))

# mkdir
from pathlib import Path
Path("./simulations_data/b/{:s}".format(date)).mkdir(parents=True, exist_ok=True)


# Hyperparameters
MOCK_PREDICTION = True
BETA = simulation_config.BETA
NUM_ARM = simulation_config.NUM_ARM
NUM_SELECTION = simulation_config.NUM_SELECTION
NUM_ROUND = simulation_config.NUM_ROUND
NUM_REPETITION = simulation_config.NUM_REPETITION
NUM_PROCESS = simulation_config.NUM_PROCESS
LATENCY_MEAN_LIST = simulation_config.LATENCY_MEAN_LIST
LATENCY_STD = simulation_config.LATENCY_STD
LATENCY_MIN = simulation_config.LATENCY_MIN
LATENCY_MAX = simulation_config.LATENCY_MAX
E_ARRAY = simulation_config.E_ARRAY
V = simulation_config.V
# V_ARRAY = np.array([0.1, 1.0, 10.0, 100.0, 1000.0])
# V_ARRAY = np.array([ 1, 10, 100, 1000, 10000, 100000 ])
# V_ARRAY = np.arange(100, 1001, 100)
# V_ARRAY[0] = 1.0
# B = simulation_config.B
B_ARRAY = np.arange(20, 35, 1)
# B_ARRAY = np.arange(12.0, 15.1, 0.5)
# B_ARRAY = np.arange(11.0, 16.1, 0.1)
# B_ARRAY = np.arange(15.0, 25.0, 1.0)
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
print("    NUM_PROCESS: {:d}".format(NUM_PROCESS))
print("    LATENCY_MEAN_LIST:", LATENCY_MEAN_LIST)
print("    LATENCY_STD: {:f}".format(LATENCY_STD))
print("    LATENCY_MIN: {:f}".format(LATENCY_MIN))
print("    LATENCY_MAX: {:f}".format(LATENCY_MAX))
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
print("{:d} emoji predictors, arm_predictor_index_mapping_list, history_information, and user_input_corpus loaded in {:s}.".format(len(predictor_list), format_time(end_time - begin_time)))


def run_agent(env, agent, filename):
    print("Started simulation for {:s}...".format(agent.name))

    begin_time = time.time()

    simulation = Simulation(env, agent, num_round=NUM_ROUND, num_repetition=NUM_REPETITION)
    simulation.run(log=True, num_process=NUM_PROCESS)

    # Throw away useless data for smaller pickled size
    simulation.prepare_dump()

    with open(filename, "wb") as f:
        pickle.dump(simulation, f)

    end_time = time.time()
    print("Finished simulation and dumped simulation to file \"{:s}\" in {:s}.".format(filename, format_time(end_time - begin_time)))


# G-LEAP agents with different values of budget
for b in B_ARRAY:
    # Build the environment
    env = EmojiPredictionEnv(
        name="EmojiPredictionEnv",
        beta=BETA,
        num_arm=NUM_ARM,
        num_selection=NUM_SELECTION,
        predictor_list=predictor_list,
        accuracy_list=accuracy_list,
        arm_predictor_index_mapping_list=arm_predictor_index_mapping_list,
        user_input_corpus=user_input_corpus,
        latency_mean_list=LATENCY_MEAN_LIST,
        latency_std=LATENCY_STD,
        latency_min=LATENCY_MIN,
        latency_max=LATENCY_MAX,
        e_array=E_ARRAY,
        b=b,
        mock_prediction=MOCK_PREDICTION)

    agent = GleapAgent(
        env,
        NUM_SELECTION,
        b,
        name="G-LEAP, $B = {:.1f}$".format(b),
        gamma=GAMMA,
        beta=BETA,
        history_information=history_information,
        v=V,
        h=H)
    filename = "./simulations_data/b/{:s}/simulation_gleap_b_{:.1f}.pkl".format(date, b)
    run_agent(env, agent, filename)


total_end_time = time.time()
print("run_b_simulations.py finished in {:s}.".format(format_time(total_end_time - total_begin_time)))
print("DATE:", date)
