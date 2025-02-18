# Reference: https://github.com/coltekin/emoji2018/blob/master/predict-linear.py

import sys
import time
total_begin_time = time.time()

from nlp.statistical.util.emoji_data import load
from nlp.statistical.util.features import doc_to_ngrams, preprocess
from util import format_time, format_binary_size


begin_time = time.time()
print("Loading and pre-processing the datasets...")
data_train = load("./dataset/semeval_2018/trial/us_trial") # TODO: Replace the trial dataset with the training dataset
data_test = load("./dataset/semeval_2018/test/us_test")

print("type(data_train.docs):", type(data_train.docs)) # <class 'list'>
print("type(data_train.docs[0]):", type(data_train.docs[0])) # <class 'str'>
for i in range(5):
    print("data_train.docs[{:d}]:".format(i), data_train.docs[i])
docs_train, v, _ = doc_to_ngrams(
    data_train.docs,
    min_df=2,
    cache_dir=".cache",
    dim_reduce=None,
    c_ngmin=1,
    c_ngmax=1,
    w_ngmin=1,
    w_ngmax=1,
    lowercase="word")
print("type(v):", type(v)) # <class 'sklearn.feature_extraction.text.TfidfVectorizer'>
print("Size of v object: {:s}".format(format_binary_size(sys.getsizeof(v))))
docs_test = preprocess(
    data_test.docs,
    c_ngmin=1,
    c_ngmax=1,
    w_ngmin=1,
    w_ngmax=1,
    lowercase="word")
print("type(docs_test):", type(docs_test)) # <class 'list'>
print("type(docs_test[0]):", type(docs_test[0])) # <class 'list'>
print("type(docs_test[0][0]):", type(docs_test[0][0])) # <class 'str'>
for i in range(5):
    print("docs_test[{:d}]: {:s}".format(i, str(docs_test[i])))
docs_test = v.transform(docs_test)
print("Size of docs_train object: {:s}".format(format_binary_size(sys.getsizeof(docs_train))))
docs_train_ndarray = docs_train.toarray()
print("docs_train_ndarray.shape:", docs_train_ndarray.shape)
print("Size of docs_train_ndarray object: {:s}".format(format_binary_size(sys.getsizeof(docs_train_ndarray))))
# print("type(docs_train):", type(docs_train)) # <class 'scipy.sparse.csr.csr_matrix'>
# print("type(docs_test):", type(docs_test)) # <class 'scipy.sparse.csr.csr_matrix'>

# Load the test labels
with open("./dataset/semeval_2018/test/us_test.labels", "r") as file:
    raw_test_label_list = file.readlines()
    test_label_list = [ int(raw_test_label.strip()) for raw_test_label in raw_test_label_list ]
    num_test_sample = len(test_label_list)

end_time = time.time()
print("Datasets loaded and pre-processed in {:s}".format(format_time(end_time - begin_time)))
print()


## Naive Bayes
begin_time = time.time()
print("### Naive Bayes")
from sklearn.naive_bayes import GaussianNB, MultinomialNB
# nb_classifier = GaussianNB() # GaussianNB is too slow, and does not accept csr_matrix as its first parameter for the fit() function
nb_classifier = MultinomialNB()

# Train
# print(type(data_train.labels)) # <class 'list'>
nb_classifier.fit(docs_train, data_train.labels)
# nb_classifier.fit(docs_train.toarray(), data_train.labels)
print("  Training time: {:s}".format(format_time(time.time() - begin_time)))

# Test
begin_time = time.time()
pred = nb_classifier.predict(docs_test)
# pred = nb_classifier.predict(docs_test.toarray())
end_time = time.time()
print("  Inference time (per sample): {:s} / {:d} = {:s}".format(format_time(end_time - begin_time), num_test_sample, format_time((end_time - begin_time) / num_test_sample)))

num_total = 0
num_correct = 0
for index, label_predicted in enumerate(pred):
    num_total += 1

    label_predicted = int(label_predicted)
    label_true = test_label_list[index]
    if label_predicted == label_true:
        num_correct += 1
test_accuracy = num_correct / num_total
print("  Test accuracy: {:d} / {:d} = {:.2f}%".format(num_correct, num_total, test_accuracy * 100))
print()


## SVM
begin_time = time.time()
print("### SVM")
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
m = LinearSVC(dual=True, C=1.0, verbose=0)
m = OneVsRestClassifier(m, n_jobs=-1)

# Train
m.fit(docs_train, data_train.labels)
print("  Training time: {:s}".format(format_time(time.time() - begin_time)))

# Test
begin_time = time.time()
pred = m.predict(docs_test)
end_time = time.time()
print("  Inference time (per sample): {:s} / {:d} = {:s}".format(format_time(end_time - begin_time), num_test_sample, format_time((end_time - begin_time) / num_test_sample)))

num_total = 0
num_correct = 0
for index, label_predicted in enumerate(pred):
    num_total += 1

    label_predicted = int(label_predicted)
    label_true = test_label_list[index]
    if label_predicted == label_true:
        num_correct += 1
test_accuracy = num_correct / num_total
print("  Test accuracy: {:d} / {:d} = {:.2f}%".format(num_correct, num_total, test_accuracy * 100))
print()


## Decision Tree
begin_time = time.time()
print("### Decision Tree")
from sklearn import tree
decision_tree_classifier = tree.DecisionTreeClassifier(random_state=42)

# Train
decision_tree_classifier.fit(docs_train, data_train.labels)
print("  Training time: {:s}".format(format_time(time.time() - begin_time)))

# Test
begin_time = time.time()
pred = decision_tree_classifier.predict(docs_test)
end_time = time.time()
print("  Inference time (per sample): {:s} / {:d} = {:s}".format(format_time(end_time - begin_time), num_test_sample, format_time((end_time - begin_time) / num_test_sample)))

num_total = 0
num_correct = 0
for index, label_predicted in enumerate(pred):
    num_total += 1

    label_predicted = int(label_predicted)
    label_true = test_label_list[index]
    if label_predicted == label_true:
        num_correct += 1
test_accuracy = num_correct / num_total
print("  Test accuracy: {:d} / {:d} = {:.2f}%".format(num_correct, num_total, test_accuracy * 100))
print()


total_end_time = time.time()
print("Total time usage: {:s}".format(format_time(total_end_time - total_begin_time)))
