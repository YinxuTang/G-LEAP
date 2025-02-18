import pickle


accuracy_list = [
    10462/50000, # SVM
    11110/50000, # Naive Bayes
    6568/50000, # Decision Tree
    12781/50000, # RNN-1
    13265/50000, # RNN-2
    14255/50000, # LSTM-1 TODO: Replace with real data
    14677/50000, # LSTM-2 TODO: Replace with real data
    15342/50000, # Bi-LSTM TODO: Replace with real data
    16532/50000, # BERT
    16441/50000  # RoBERTa
]
with open("accuracy_list.pkl", "wb") as file:
    pickle.dump(accuracy_list, file, protocol=5)


arm_predictor_index_mapping_list = [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ]
with open("arm_predictor_index_mapping_list.pkl", "wb") as file:
    pickle.dump(arm_predictor_index_mapping_list, file, protocol=5)
