from sklearn import tree

from . import StatisticalEmojiPredictor


class DecisionTreeEmojiPredictor(StatisticalEmojiPredictor):
    def __init__(self, vectorizer, name="DecisionTreeEmojiPredictor", **kwds):
        # Create a Decision Tree classifier
        classifier = tree.DecisionTreeClassifier(random_state=42)

        super().__init__(classifier, vectorizer, name=name, **kwds)
