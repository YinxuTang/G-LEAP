class Training(object):
    __slots__ = (
        "neural_emoji_predictor",
        "training_dataset",
        "validation_dataset",
        "learning_rate",
        "momentum",
        "num_epoch",
        "label",
        "training_progress_dict")


    def __init__(self, neural_emoji_predictor, training_dataset, validation_dataset, learning_rate, momentum, num_epoch, label):
        self.neural_emoji_predictor = neural_emoji_predictor
        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.num_epoch = num_epoch
        self.label = label


    def run(self):
        self.training_progress_dict = self.neural_emoji_predictor.train(
            self.training_dataset,
            validation_dataset=self.validation_dataset,
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            num_epoch=self.num_epoch)
    

    def prepare_dump(self):
        self.training_dataset = None
        self.validation_dataset = None
        self.neural_emoji_predictor = str(type(self.neural_emoji_predictor))
