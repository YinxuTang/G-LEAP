import time

import numpy as np

from util import format_time


class Vectorizer(object):
    __slots__ = ("vector_length")


    def __init__(self):
        pass


    def vectorize_sequence(self, sequence):
        pass


    def vectorize_word(self, word):
        pass


class TfidfVectorizer(Vectorizer):
    def __init__(self):
        # TODO: Implementation
        pass


class GloveVectorizer(Vectorizer):
    """Vectorizer using GloVe embeddings and mean values of each word vectorers.
    """

    __slots__ = ( "glove_word2vec_dict" )


    def __init__(self, glove_embeddings_path="./glove_embeddings/glove.twitter.27B.25d.txt"):
        super().__init__()

        # Load the GloVe embeddings
        begin_time = time.time()
        self.glove_word2vec_dict = {}
        with open(glove_embeddings_path, "r") as file:
            for line in file:
                item_list = line.split()

                # Extract the word
                word = item_list[0]
                
                # Extract the vector
                vector_length = len(item_list) - 1
                vector = np.zeros(vector_length, dtype=float)
                for i in range(vector_length):
                    vector[i] = float(item_list[i + 1])

                # Store the word, vector pair into the dictionary
                self.glove_word2vec_dict[word] = vector
        end_time = time.time()
        self.vector_length = vector_length
        print("GloveVectorizer: {:d} word vectors of length {:d} loaded in {:s}".format(
            len(self.glove_word2vec_dict.keys()),
            self.vector_length,
            format_time(end_time - begin_time)))
    

    def vectorize_sequence(self, sequence):
        # Tokenize the sentence
        word_list = sequence.split()

        # Use mean (averaging) as the sentence embeddings
        vector_list = []
        for word in word_list:
            if word in self.glove_word2vec_dict:
                vector_list.append(self.glove_word2vec_dict[word])
        if len(vector_list) > 0:
            mean_vector = np.mean(np.array(vector_list, dtype=float), axis=0)
        else:
            mean_vector = np.zeros(self.vector_length, dtype=float)

        return mean_vector
    

    def vectorize_word(self, word):
        # If the word is in the word embeddings dictionary
        if word in self.glove_word2vec_dict:
            return self.glove_word2vec_dict[word]
        # Else, out of vocalbulary (OOV)
        else:
            return np.zeros(self.vector_length, dtype=float)
