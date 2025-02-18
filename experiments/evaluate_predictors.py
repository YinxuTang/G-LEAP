"""
"""


import csv
import datetime
import pickle
import time
total_begin_time = time.time()

import numpy as np
import torch
import torch.nn as nn

from nlp.dataset_utils import load_dataset
from util import format_time, format_now


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print('Using PyTorch version:', torch.__version__, ' Device:', device)


# # Load the predictor list
# with open("./emoji_predictor_list.pkl", "rb") as f:
#     emoji_predictor_list = pickle.load(f)
#     print("{:d} emoji predictors loaded".format(len(emoji_predictor_list)))


# Load the test dataset
test_dataset = load_dataset("./dataset/semeval_2018/trial/us_trial") # TODO: Use test dataset for evaluating all the emoji predictors.
print()


# Load the transformer emoji predictors
emoji_predictor_list = []
with open("./trained_emoji_predictors/neural/transformer/bert.pkl", "rb") as file:
    bert_emoji_predictor = pickle.load(file)
    emoji_predictor_list.append(bert_emoji_predictor)
with open("./trained_emoji_predictors/neural/transformer/roberta.pkl", "rb") as file:
    roberta_emoji_predictor = pickle.load(file)
    emoji_predictor_list.append(roberta_emoji_predictor)


# Evaluate the predictors one by one
# # accuracy_list = []
# accuracy_list = [ 0, 0, 0 ] # TODO: Evaluate the first 8 models and deprecate mock_prediction
for predictor in emoji_predictor_list:
    print("\nEvaluating {:s}...".format(predictor.name))

    # Conduct inference on GPU if possible for acceleration
    predictor.to_device(device)

    # For logging
    num_test_sample = len(test_dataset["text_list"])
    iteration_time_consumption_array = np.zeros(num_test_sample, dtype=float)
    log_interval = num_test_sample // 100
    total_begin_datetime = datetime.datetime.now()
    num_digit = 0
    temp = num_test_sample
    while temp > 1:
        temp /= 10
        num_digit += 1
    print("  Evaluation started at: {:s}".format(total_begin_datetime.strftime("%Y-%m-%d %H:%M:%S")))

    num_correct = 0
    for sample_index, text in enumerate(test_dataset["text_list"]):
        # For logging
        begin_time = time.time()

        true_label = test_dataset["label_list"][sample_index]
        predicted_label = int(predictor.predict(text, top_k=1)[0])
        if true_label == predicted_label:
            num_correct += 1
        
        # For logging
        end_time = time.time()
        iteration_time_consumption = end_time - begin_time
        iteration_time_consumption_array[sample_index] = iteration_time_consumption
        average_time_consumption = np.mean(iteration_time_consumption_array[:sample_index + 1])
        eta_datetime = total_begin_datetime + datetime.timedelta(seconds=average_time_consumption * num_test_sample)
        if (sample_index + 1) % log_interval == 0:
            if sample_index < num_test_sample - 1:
                eta_datetimestr = eta_datetime.strftime("%Y-%m-%d %H:%M:%S")
                template = "  {:6.2f}% {:" + str(num_digit) + "d} samples finished in {:>10s}. Average: {:>10s}/iter. ETA: {:s}."
                print(template.format(
                    (sample_index + 1) * 100 / num_test_sample,
                    sample_index + 1,
                    format_time(iteration_time_consumption),
                    format_time(average_time_consumption),
                    eta_datetimestr))
            else:
                completion_datetimestr = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                template = "  {:6.2f}% {:" + str(num_digit) + "d} samples finished in {:>10s}. Average: {:>10s}/iter. Finished at: {:s}."
                print(template.format(
                    (sample_index + 1) * 100 / num_test_sample,
                    sample_index + 1,
                    format_time(iteration_time_consumption),
                    format_time(average_time_consumption),
                    completion_datetimestr))
    accuracy = num_correct / num_test_sample
    print("Accuracy of {:s}: {:d}/{:d} = {:6.2f}%".format(predictor.name, num_correct, num_test_sample, round(accuracy * 100, 2)))


# # Dump the accuracy list
# filename = "./accuracy_list.pkl"
# with open(filename, "wb") as file:
#     pickle.dump(accuracy_list, file)
#     print("\nThe accuracy list has been dumped to the file \"{:s}\".".format(filename))


total_end_time = time.time()
print("{:s}: evaluate_predictors.py finished in {:s}.".format(format_now(), format_time(total_end_time - total_begin_time)))
