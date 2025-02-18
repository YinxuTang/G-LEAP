"""Prepares the statistical models for emoji prediction.

- SVM
- Naive Bayes
- Decision Tree

For each statistical model:
1. Create and train a model on the dataset.
2. Persist the trained model to the file system (`./simulation_data/prepare_statistical/${DATE}/predictor/*.pkl`).
"""


import argparse
from pathlib import Path
import pickle
import shutil
import sys
import time
total_begin_time = time.time()
import copy

from nlp.statistical import SvmEmojiPredictor, NaiveBayesEmojiPredictor, DecisionTreeEmojiPredictor
from nlp.statistical.util.features import doc_to_ngrams
from nlp.dataset_utils import load_dataset
from nlp.vectorizer import GloveVectorizer
from util import format_time, format_now, format_binary_size


# Hyperparameters
TRAINING_DATASET_SIZE = 5000


# For parsing command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--date', type=str, help='The date (time) when the script start.')
args = parser.parse_args()
DATE = args.date
print("prepare_statistical.py: got argument date: {:s}".format(DATE))


# The directory for storing the trained models
MODEL_DIRECTORY_PATH = "./simulations_data/prepare_statistical/{:s}/predictor/".format(DATE)
# Create the directory if not exists (mkdir -p)
Path(MODEL_DIRECTORY_PATH).mkdir(parents=True, exist_ok=True)


# Load the datasets
begin_time = time.time()
print("Loading the datasets...")
training_dataset = load_dataset("./dataset/semeval_2018/train/us_train", size=TRAINING_DATASET_SIZE)
print("Size of training_dataset: {:s}".format(format_binary_size(sys.getsizeof(training_dataset))))
validation_dataset = load_dataset("./dataset/semeval_2018/trial/us_trial")
end_time = time.time()
print("Datasets loaded in {:s}".format(format_time(end_time - begin_time)))
print()


# Construct the TF-IDF vectorizer
_, tfidf_vectorizer, _ = doc_to_ngrams(
    copy.deepcopy(training_dataset["text_list"]),
    min_df=2,
    cache_dir=".cache",
    dim_reduce=None,
    c_ngmin=1,
    c_ngmax=1,
    w_ngmin=1,
    w_ngmax=1,
    lowercase="word")


# Construct the GloVe embeddings vectorizer
glove_vectorizer = GloveVectorizer("./glove_embeddings/glove.twitter.27B.50d.txt")
print()


def train_and_validate_predictor(predictor, training_dataset, validation_dataset, filename):
    # Train
    print("  Size (before training): {:s}".format(format_binary_size(sys.getsizeof(predictor))))
    begin_time = time.time()
    predictor.train(training_dataset)
    end_time = time.time()
    print("  Training time: {:s}".format(format_time(end_time - begin_time)))
    # Persist the model
    print("  Size (after training): {:s}".format(format_binary_size(sys.getsizeof(predictor))))
    with open(filename, "wb") as file:
        pickle.dump(predictor, file, protocol=5)
        print("  The predictor \"{:s}\" has been dumped to the file \"{:s}\"".format(predictor.name, filename))
    print()

    # Validate
    predicted_results = predictor.predict(validation_dataset["text_list"])
    num_total = 0
    num_correct = 0
    for index, label_predicted in enumerate(predicted_results):
        num_total += 1

        label_predicted = int(label_predicted)
        label_true = validation_dataset["label_list"][index]
        if label_predicted == label_true:
            num_correct += 1
    validation_accuracy = num_correct / num_total
    print("  Validation accuracy: {:d} / {:d} = {:.2f}%".format(num_correct, num_total, validation_accuracy * 100))
    print()


# SVM with TF-IDF
print("### SVM with TF-IDF")
# Create the predictor
svm_emoji_predictor = SvmEmojiPredictor(copy.deepcopy(tfidf_vectorizer), name="SVM with TF-IDF")
train_and_validate_predictor(
    svm_emoji_predictor,
    copy.deepcopy(training_dataset),
    copy.deepcopy(validation_dataset),
    MODEL_DIRECTORY_PATH + "svm_with_tfidf.pkl")


# Naive Bayes with TF-IDF
print("### Naive Bayes with TF-IDF")
# Create the predictor
naive_bayes_emoji_predictor = NaiveBayesEmojiPredictor(copy.deepcopy(tfidf_vectorizer), name="Naive Bayes with TF-IDF")
train_and_validate_predictor(
    naive_bayes_emoji_predictor,
    copy.deepcopy(training_dataset),
    copy.deepcopy(validation_dataset),
    MODEL_DIRECTORY_PATH + "naive_bayes_with_tfidf.pkl")


# Decision Tree with TF-IDF
print("### Decision Tree with TF-IDF")
# Create the predictor
decision_tree_emoji_predictor = DecisionTreeEmojiPredictor(copy.deepcopy(tfidf_vectorizer), name="Decision Tree with TF-IDF")
train_and_validate_predictor(
    decision_tree_emoji_predictor,
    copy.deepcopy(training_dataset),
    copy.deepcopy(validation_dataset),
    MODEL_DIRECTORY_PATH + "decision_tree_with_tfidf.pkl")


# SVM with GloVe embeddings
print("### SVM with GloVe")
# Create the predictor
svm_emoji_predictor = SvmEmojiPredictor(copy.deepcopy(glove_vectorizer), name="SVM with GloVe")
train_and_validate_predictor(
    svm_emoji_predictor,
    copy.deepcopy(training_dataset),
    copy.deepcopy(validation_dataset),
    MODEL_DIRECTORY_PATH + "svm_with_glove.pkl")


# Naive Bayes with GloVe embeddings
print("### Naive Bayes with GloVe")
# Create the predictor
naive_bayes_emoji_predictor = NaiveBayesEmojiPredictor(copy.deepcopy(glove_vectorizer), name="Naive Bayes with GloVe")
train_and_validate_predictor(
    naive_bayes_emoji_predictor,
    copy.deepcopy(training_dataset),
    copy.deepcopy(validation_dataset),
    MODEL_DIRECTORY_PATH + "naive_bayes_with_glove.pkl")


# Decision Tree with GloVe embeddings
print("### Decision Tree with GloVe")
# Create the predictor
decision_tree_emoji_predictor = DecisionTreeEmojiPredictor(copy.deepcopy(glove_vectorizer), name="Decision Tree with GloVe")
train_and_validate_predictor(
    decision_tree_emoji_predictor,
    copy.deepcopy(training_dataset),
    copy.deepcopy(validation_dataset),
    MODEL_DIRECTORY_PATH + "decision_tree_with_glove.pkl")


total_end_time = time.time()
print("{:s}: prepare_statistical.py finished in {:s}.".format(format_now(), format_time(end_time - begin_time)))
print("DATE: {:s}".format(DATE))
