import argparse
import copy
import math
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
print("Running ht simulations for start date: {:s}".format(date))

# mkdir
from pathlib import Path
Path("./simulations_data/ht/{:s}".format(date)).mkdir(parents=True, exist_ok=True)


# Hyperparameters
MOCK_PREDICTION = True
BETA = simulation_config.BETA
NUM_ARM = simulation_config.NUM_ARM
NUM_SELECTION = simulation_config.NUM_SELECTION
# NUM_ROUND = simulation_config.NUM_ROUND
# NUM_ROUND_ARRAY = np.array([ 20000, 40000, 60000, 80000, 100000 ])
# NUM_ROUND_ARRAY = np.array([ 200, 400, 600, 800, 1000 ])
NUM_ROUND_ARRAY = np.array([ 2000, 4000, 6000, 8000, 10000 ])
NUM_REPETITION = simulation_config.NUM_REPETITION
NUM_PROCESS = simulation_config.NUM_PROCESS
LATENCY_MEAN_LIST = simulation_config.LATENCY_MEAN_LIST
LATENCY_STD = simulation_config.LATENCY_STD
LATENCY_MIN = simulation_config.LATENCY_MIN
LATENCY_MAX = simulation_config.LATENCY_MAX
E_ARRAY = simulation_config.E_ARRAY
V = simulation_config.V
B = simulation_config.B
# H = simulation_config.H
H_FUNCTION_LIST = [ lambda t: 0, lambda t: int(round(0.1 * t)), lambda t: int(t), lambda t: int(round(t * math.log(t))) ]
GAMMA = simulation_config.GAMMA


# Print the setting of the hyperparameters
print("---------Hyperparameters----------")
print("    MOCK_PREDICTION:", MOCK_PREDICTION)
print("    BETA: {:f}".format(BETA))
print("    NUM_ARM: {:d}".format(NUM_ARM))
print("    NUM_SELECTION: {:d}".format(NUM_SELECTION))
print("    NUM_ROUND_ARRAY:", NUM_ROUND_ARRAY)
print("    NUM_REPETITION: {:d}".format(NUM_REPETITION))
print("    NUM_PROCESS: {:d}".format(NUM_PROCESS))
print("    LATENCY_MEAN_LIST:", LATENCY_MEAN_LIST)
print("    E_ARRAY:", E_ARRAY)
print("    V: {:.1f}".format(V))
print("    B: {:.1f}".format(B))
print("    H_FUNCTION_LIST:", H_FUNCTION_LIST)
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
print("{:d} inference models, arm_predictor_index_mapping_list, history_information and user_input_corpus loaded in {:s}.".format(len(predictor_list), format_time(end_time - begin_time)))


def run_agent(env, agent, filename, num_round, h_function_index):
    print("Started simulation for {:s}...".format(agent.name))

    begin_time = time.time()

    simulation = Simulation(env, agent, num_round=num_round, num_repetition=NUM_REPETITION)
    simulation.h_function_index = h_function_index
    simulation.run(log=True, num_process=NUM_PROCESS)

    # Throw away useless data for smaller pickled size
    simulation.prepare_dump()

    with open(filename, "wb") as f:
        pickle.dump(simulation, f)

    end_time = time.time()
    print("Finished simulation and dumped simulation to file \'{:s}\' in {:s}".format(filename, format_time(end_time - begin_time)))


for num_round in NUM_ROUND_ARRAY:
    for h_function_index, h_function in enumerate(H_FUNCTION_LIST):
        # Build the environment
        env = EmojiPredictionEnv(name="EmojiPredictionEnv", beta=BETA, num_arm=NUM_ARM, num_selection=NUM_SELECTION, predictor_list=predictor_list, accuracy_list=accuracy_list, arm_predictor_index_mapping_list=arm_predictor_index_mapping_list, user_input_corpus=user_input_corpus, latency_mean_list=LATENCY_MEAN_LIST, e_array=E_ARRAY, b=B, mock_prediction=MOCK_PREDICTION)
        
        # Create the G-LEAP agent
        h = h_function(num_round)
        agent = GleapAgent(env, NUM_SELECTION, B, name="G-LEAP, $H = {:d}$, $T = {:d}$".format(h, num_round), gamma=GAMMA, beta=BETA, history_information=history_information, v=V, h=h)
        
        # Run the agent in the environment
        filename = "./simulations_data/ht/{:s}/simulation_gleap_h_{:d}_num_round_{:d}.pkl".format(date, h, num_round)
        run_agent(env, agent, filename, num_round, h_function_index)


total_end_time = time.time()
print("All ht simulations finished in {:s}.".format(format_time(total_end_time - total_begin_time)))
print("DATE:", date)
