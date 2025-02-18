from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

from . import StatisticalEmojiPredictor


class SvmEmojiPredictor(StatisticalEmojiPredictor):
    def __init__(self, vectorizer, name="SvmEmojiPredictor", **kwds):
        # Create an SVM classifier
        classifier = OneVsRestClassifier(
            LinearSVC(dual=True, C=1.0, verbose=0),
            n_jobs=-1)
        
        super().__init__(classifier, vectorizer, name=name, **kwds)
