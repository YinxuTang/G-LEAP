import pickle

import numpy as np
from scipy.special import softmax

from nlp.neural import NeuralEmojiPredictor


class TransformerEmojiPredictor(NeuralEmojiPredictor):
    # # Load the label index mapping list from the cardiffnlp transformer models to the dataset
    # with open("./dataset/cardiffnlp_to_dataset_label_index_mapping_list.pkl", "rb") as f:
    #     _cardiffnlp_to_dataset_label_index_mapping_list = pickle.load(f)

    def __init__(self, tokenizer, language_model, name="TransformerEmojiPredictor", **kwds):
        super().__init__(name=name, **kwds)

        self._tokenizer = tokenizer
        self._language_model = language_model
        self._device = 'cpu'

    
    def predict(self, text, top_k=3):
        # Preprocess the text
        text = self._preprocess_text(text)

        # Encode the input
        encoded_input = self._tokenizer(text, return_tensors='pt')
        encoded_input.to(self._device)

        # Run the model and get the inferred result
        output = self._language_model(**encoded_input)
        scores = output[0][0].detach().cpu().numpy()
        scores = softmax(scores)

        # Sort the labels in descending order of the score
        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        
        # Construct the result list, i.e., the list of the predicted emojis
        result_list = []
        for i in range(top_k):
            # label = self._cardiffnlp_to_dataset_label_index_mapping_list[ranking[i]]
            label = ranking[i]
            score = scores[ranking[i]]
            result_list.append(label)
        
        return result_list
    

    def _preprocess_text(self, text):
        """Preprocesses the text (for username and link placeholders).

        Arguments:
            text {str} -- The text.

        Returns:
            [str] -- The preprocessed text.
        """

        new_text = []
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)
    

    def to_device(self, device):
        self._device = device
        self._language_model.to(self._device)
