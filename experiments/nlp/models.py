class EmojiPredictor:
    def __init__(self, name="EmojiPredictor"):
        self.name = name
    

    def train(self, training_dataset):
        """Trains this emoji predictor on the given dataset.

        Arguments:
            dataset {[type]} -- [description] TODO: Adds the documentation.
        """
        pass
    

    def predict(self, text, top_k=1):
        """Predicts proper emojis associated with the text.

        Arguments:
            text {str} or {TODO: specify the type for batch prediction} -- The input text.

        Keyword Arguments:
            top_k {int} -- The number of emojis to be predicted. (default: {1})
        """
        pass
