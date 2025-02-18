import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .. import EmojiPredictor
from nlp.statistical.util.features import preprocess


class StatisticalEmojiPredictor(EmojiPredictor):
    __slots__ = ("vectorizer", "classifier")


    def __init__(self, classifier, vectorizer, **kwds):
        super().__init__(**kwds)

        self.classifier = classifier
        self.vectorizer = vectorizer
    

    def train(self, training_dataset):
        text_list = training_dataset["text_list"]
        label_list = training_dataset["label_list"]

        # TF-IDF vectorizer (sparse vectors)
        if isinstance(self.vectorizer, TfidfVectorizer):
            x = preprocess(
                text_list,
                c_ngmin=1,
                c_ngmax=1,
                w_ngmin=1,
                w_ngmax=1,
                lowercase="word")
            x = self.vectorizer.transform(x)
        # Word Embeddings (dense vectors)
        else:
            num_sequence = len(text_list)
            x = np.zeros((num_sequence, self.vectorizer.vector_length), dtype=float)
            for i, sequence in enumerate(text_list):
                x[i] = self.vectorizer.vectorize_sequence(sequence)

        self.classifier.fit(x, label_list)
    

    def predict(self, text):
        """[summary] TODO: Add documentation.

        Arguments:
            text {str or list of str} -- The input text.

        Returns:
            [type] -- [description]
        """

        single_input = False
        if isinstance(text, str):
            single_input = True
            text = [ text ]
        
        # TF-IDF vectorizer (sparse vectors)
        if isinstance(self.vectorizer, TfidfVectorizer):
            x = preprocess(
                text,
                c_ngmin=1,
                c_ngmax=1,
                w_ngmin=1,
                w_ngmax=1,
                lowercase="word")
            x = self.vectorizer.transform(x)
        # Word Embeddings (dense vectors)
        else:
            num_sequence = len(text)
            x = np.zeros((num_sequence, self.vectorizer.vector_length), dtype=float)
            for i, sequence in enumerate(text):
                x[i] = self.vectorizer.vectorize_sequence(sequence)

        predicted_result = self.classifier.predict(x)
        # print("type(predicted_result):", type(predicted_result)) # <class 'numpy.ndarray'>
        if single_input:
            predicted_result = predicted_result[0]
        return predicted_result
