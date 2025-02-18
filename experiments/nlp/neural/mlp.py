"""MLP emoji predictors.

Reference (conv2d layers removed): https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from nlp.neural.dataset_utils import EmojiPredictionDataset
from nlp.neural import NeuralEmojiPredictor
from nlp.vectorizer import GloveVectorizer


class Mlp(nn.Module):
    r"""Multi layer perceptron (MLP) with ReLU as the activation function."""

    __slots__ = ("name", "input_dimension", "output_dimension", "hidden_layer_dimension_list", "hidden_layers", "output_layer")


    def __init__(self, input_dimension, output_dimension, hidden_layer_dimension_list, linear_layer=torch.nn.Linear, name="MLP"):
        """ Constructor.
        Args:
            input_dimension: int. The dimension of input.
            output_dimension: int. The dimension of output. Usually the number of categories for a classification problem.
            hidden_layer_diemnsion_list: list of int. Each number in the list represents the number of neurons in the corresponding hidden layer.
            linear_layer: torch.nn.Linear or its subclass.
            name: str. The name of this MLP.
        """
        super().__init__()
        
        self.name = name
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.hidden_layer_dimension_list = hidden_layer_dimension_list
        
        # Layer definitions
        self.hidden_layers = nn.ModuleList([])
        for index, dimension in enumerate(self.hidden_layer_dimension_list):
            # Get the dimension of the previous layer
            previous_dimension = self.input_dimension if index == 0 else self.hidden_layer_dimension_list[index - 1]

            # Define the current hidden layer
            self.hidden_layers.append(linear_layer(previous_dimension, dimension))
        self.output_layer = nn.Linear(self.hidden_layer_dimension_list[-1], self.output_dimension)


    def forward(self, x):
        x = x.view(-1, self.input_dimension)
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        return F.log_softmax(self.output_layer(x), dim=1)
        return x


class VectorizationTransform(object):
    __slots__ = ("vectorizer")


    def __init__(self, vectorizer):
        self.vectorizer = vectorizer


    def __call__(self, sample):
        text_array, y = sample["x"], sample["y"]
        # print("text_array:", text_array)
        # print("type(text_array):", type(text_array)) # <class 'numpy.ndarray'>
        # print("text_array.shape:", text_array.shape)
        # print("label_array:", label_array)

        # if len(text_array.shape) > 0:
        #     x = np.zeros(text_array.shape, dtype=float)
        #     for text_index, text in enumerate(text_array):
        #         x[text_index] = self.vectorizer.vectorize(text)
        # # Else, text_array is only a single sample
        # else:
        #     text = str(text_array)
        #     x = self.vectorizer.vectorize(text)

        # The text array is only a single sentence (text)
        text = str(text_array)
        x = self.vectorizer.vectorize_sequence(text)

        return { "x": torch.from_numpy(x), "y": torch.from_numpy(y) }


class MlpEmojiPredictor(NeuralEmojiPredictor):
    __slots__ = ("vectorizer", "model")


    def __init__(self, vectorizer=None, model=None, name="MlpEmojiPredictor", **kwds):
        super().__init__(name=name, **kwds)

        # For the vectorizer
        self.vectorizer = vectorizer
        if self.vectorizer is None:
            self.vectorizer = GloveVectorizer()
        
        # For the model
        self.model = model
        if self.model is None:
            self.model = Mlp(self.vectorizer.vector_length, 20, [ 32, 64, 32 ])


    def train(self, training_dataset, validation_dataset=None, device=None, learning_rate=1e-3, num_epoch=100):
        # If device is not set, then use GPU if available
        if device is None:
            if torch.cuda.is_available():
                device = torch.device('cuda')
                print("GPU is available. Using GPU.")
            else:
                device = torch.device('cpu')
        print("device:", device)

        # Move the model to the device
        self.model.to(device)

        # Create the DataLoader
        training_dataset_torch = EmojiPredictionDataset(training_dataset, transform=VectorizationTransform(self.vectorizer))
        training_data_loader = DataLoader(training_dataset_torch, batch_size=32, shuffle=False)
        if validation_dataset:
            validation_dataset_torch = EmojiPredictionDataset(validation_dataset, transform=VectorizationTransform(self.vectorizer))
            validation_data_loader = DataLoader(validation_dataset_torch, batch_size=32, shuffle=False)

        # Define the optimizer and loss function
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.5)
        criterion = nn.NLLLoss()

        # The loss and accuracy array for this training
        training_loss_array, training_accuracy_array = np.zeros(num_epoch, dtype=float), np.zeros(num_epoch, dtype=float)
        if validation_dataset:
            validation_loss_array, validation_accuracy_array = np.zeros(num_epoch, dtype=float), np.zeros(num_epoch, dtype=float)
        else:
            validation_loss_array = None
            validation_accuracy_array = None
        
        # Train and validate for a number of epochs
        for epoch_index in range(num_epoch):
            print("Epoch #{:d}:".format(epoch_index))

            # Train
            self._train_model(
                self.model,
                optimizer,
                criterion,
                training_data_loader,
                epoch_index,
                training_loss_array,
                training_accuracy_array,
                device,
                log_interval=1000)

            # Validate if necessary
            if validation_dataset:
                self._validate_model(
                    self.model,
                    criterion,
                    validation_data_loader,
                    epoch_index,
                    validation_loss_array,
                    validation_accuracy_array,
                    device)
        
        return {
            "training_loss_array": training_loss_array,
            "training_accuracy_array": training_accuracy_array,
            "validation_loss_array": validation_loss_array,
            "validation_accuracy_array": validation_accuracy_array
        }


    def predict(self, text):
        # TODO: Implementation
        pass


    def _train_model(
        self,
        model,
        optimizer,
        criterion,
        training_data_loader,
        epoch_index,
        training_loss_array,
        training_accuracy_array,
        device,
        log_interval=100):
        # Set the model to the training mode
        model.train()
        
        # Loop over each batch sampled from the training dataset
        total_loss = 0.0
        total_num_correct = 0
        print("Training:")
        for batch_index, sample_batch in enumerate(training_data_loader):
            batch_size = sample_batch["x"].size()[0]

            # data = sample_batch["x"]
            # target = sample_batch["y"]
            data = sample_batch["x"].float()
            # print("type(sample_batch[\"y\"]):", type(sample_batch["y"]))
            # print("sample_batch[\"y\"]:", sample_batch["y"])
            target = sample_batch["y"]

            # Copy data to GPU if needed
            data = data.to(device)
            target = target.to(device)

            # Zero gradient buffers
            optimizer.zero_grad()
            
            # Inference
            output = model(data)
            pred = output.data.max(1)[1] # Get the index of the max log-probability
            total_num_correct += pred.eq(target.data).cpu().sum()

            # Calculate the loss
            loss = criterion(output, target)
            total_loss += loss.data.item()

            # Backpropagate
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Log
            if batch_index % log_interval == 0:
                print('Batch #{}: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(batch_index, batch_index * batch_size, len(training_data_loader.dataset),
                    100. * batch_index / len(training_data_loader), loss.data.item()))
        
        training_loss_array[epoch_index] = total_loss / len(training_data_loader)
        training_accuracy_array[epoch_index] = total_num_correct.to(torch.float32) / len(training_data_loader.dataset)
        print("Training:   loss: {:.4f}, accuracy: {:d}/{:d} = {:.4f}%".format(
            training_loss_array[epoch_index],
            total_num_correct.to(torch.int32),
            len(training_data_loader.dataset),
            training_accuracy_array[epoch_index] * 100))


    def _validate_model(
        self,
        model,
        criterion,
        validation_data_loader,
        epoch_index,
        validation_loss_array,
        validation_accuracy_array,
        device):
        # Set the model to the evaluation mode
        model.eval()

        val_loss, correct = 0, 0
        for sample_batch in validation_data_loader:
            # data = sample_batch["x"]
            # target = sample_batch["y"]
            data = sample_batch["x"].float()
            target = sample_batch["y"]

            # Copy data to GPU if needed
            data = data.to(device)
            target = target.to(device)

            # Inference
            output = model(data)

            val_loss += criterion(output, target).data.item()
            pred = output.data.max(1)[1] # Get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum()

        val_loss /= len(validation_data_loader)
        validation_loss_array[epoch_index] = val_loss

        accuracy = correct.to(torch.float32) / len(validation_data_loader.dataset)
        validation_accuracy_array[epoch_index] = accuracy
        
        print('Validation: loss: {:.4f}, accuracy: {:d}/{:d} = {:.4f}%\n'.format(
            val_loss, correct, len(validation_data_loader.dataset), accuracy * 100.0))
