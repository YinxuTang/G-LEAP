import numpy as np


# Hyperparameters
BETA = 0.7
# BETA = 0.7
NUM_ARM = 10
NUM_SELECTION = 3
# NUM_ROUND = 10000 # for debugging
NUM_ROUND = 100000
NUM_REPETITION = 100
# NUM_REPETITION = 1
NUM_PROCESS = NUM_REPETITION if NUM_REPETITION < 50 else 50
# LATENCY_MEAN_LIST = [ 300, 800, 1200, 800, 500, 970, 980, 800, 1130, 860 ]
LATENCY_MEAN_LIST = [ 30, 80, 120, 80, 50, 97, 98, 80, 113, 86 ]
LATENCY_STD = 20
LATENCY_MIN = 0
LATENCY_MAX = 150
E_ARRAY = np.array([ 7.4, 8.7, 10.2, 9.5, 10.8, 9.5, 8.1, 11.2, 13.3, 8.6 ], dtype=float)
V = 100000.0
B = 28.0
H_MAX = 200 * 10000
H = 10
# GAMMA = 0.8
GAMMA = 2.0
EPSILON = 0.2

ALG_NAME = 'G-LEAP'


# The filename list of the pre-trained emoji predictors
EMOJI_PREDICTOR_FILENAME_LIST = [
    "./trained_emoji_predictors/statistical/svm_with_tfidf.pkl",
    "./trained_emoji_predictors/statistical/svm_with_glove.pkl",
    "./trained_emoji_predictors/statistical/naive_bayes_with_tfidf.pkl",
    "./trained_emoji_predictors/statistical/naive_bayes_with_glove.pkl",
    "./trained_emoji_predictors/statistical/decision_tree_with_tfidf.pkl",
    "./trained_emoji_predictors/statistical/decision_tree_with_glove.pkl",
    "./trained_emoji_predictors/neural/mlp/mlp_1.pkl",
    "./trained_emoji_predictors/neural/mlp/mlp_2.pkl",
    "./trained_emoji_predictors/neural/mlp/mlp_3.pkl",
    "./trained_emoji_predictors/neural/rnn/rnn_1.pkl",
    "./trained_emoji_predictors/neural/rnn/rnn_2.pkl",
    "./trained_emoji_predictors/neural/rnn/rnn_3.pkl"
]
