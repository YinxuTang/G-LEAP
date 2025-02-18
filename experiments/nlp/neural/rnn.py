"""RNN emoji predictors.

Reference: 
"""


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from nlp.neural.dataset_utils import EmojiPredictionDataset
from nlp.neural import NeuralEmojiPredictor
from nlp.vectorizer import GloveVectorizer


# class Rnn(nn.Module):
#     """Vanilla RNN from scratch. As this is too slow compared to using nn.RNN, we do not use this class. Reference: https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
#     """


#     __slots__ = ("name", "input_dimension", "output_dimension", "hidden_dimension_list", "hidden_layers", "output_layer")


#     def __init__(self, input_dimension, output_dimension, hidden_dimension_list, name="RNN"):
#         super().__init__()

#         self.name = name
#         self.input_dimension = input_dimension
#         self.output_dimension = output_dimension
#         self.hidden_dimension_list = hidden_dimension_list

#         # Create the hidden layers
#         self.hidden_layers = nn.ModuleList([])
#         for hidden_layer_index, hidden_dimension in enumerate(self.hidden_dimension_list):
#             # For the in dimension
#             if hidden_layer_index > 0:
#                 in_dimension = self.hidden_dimension_list[hidden_layer_index]
#             else:
#                 in_dimension = self.input_dimension + self.hidden_dimension_list[hidden_layer_index]

#             # For the out dimension
#             if hidden_layer_index == len(hidden_dimension_list) - 1:
#                 out_dimension = self.hidden_dimension_list[0]
#             else:
#                 out_dimension = self.hidden_dimension_list[hidden_layer_index + 1]

#             # Create the hidden layer with (in_dimension, out_dimension)
#             self.hidden_layers.append(nn.Linear(in_dimension, out_dimension))
        
#         # Create the output layer
#         self.output_layer = nn.Linear(self.input_dimension + self.hidden_dimension_list[-1], self.output_dimension)


#     def forward(self, input, hidden):
#         combined = torch.cat((input, hidden), 1)
        
#         # Compute the hidden
#         # hidden = F.relu(self.i2h(combined))
#         hidden = combined
#         for hidden_layer_index, hidden_layer in enumerate(self.hidden_layers):
#             hidden = F.relu(hidden_layer(hidden))

#         # Compute the output
#         output = F.relu(self.output_layer(combined))
#         output = F.log_softmax(output, dim=1)
        
#         return output, hidden


#     def reset_hidden(self, device, batch_size):
#         return torch.zeros(batch_size, self.hidden_dimension_list[0]).to(device)


class Rnn(nn.Module):
    # TODO: Do not define slots, otherwise the pickle/PyTorch load(), save() will fail due to some dictionary error
    # __slots__ = ("name", "input_dimension", "output_dimension", "hidden_dimension", "hidden_layer", "output_layer")


    def __init__(self, input_dimension, output_dimension, hidden_dimension, name="RNN"):
        super().__init__()

        self.name = name
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.hidden_dimension = hidden_dimension

        # Create the hidden RNN layer
        self.hidden_layer = nn.RNN(self.input_dimension, self.hidden_dimension)

        # Create the output layer
        self.output_layer = nn.Linear(self.hidden_dimension, self.output_dimension)
    

    def forward(self, input):
        num_word = input.size()[0]
        batch_size = input.size()[1]
        # print("input.size():", input.size()) # num_word, batch_size, input_dimension
        # print("num_word:", num_word)
        # print("batch_size:", batch_size)

        # Compute the hidden
        hidden = self.hidden_layer(input)[1][0].view(batch_size, -1)

        # hidden = self.hidden_layer(input)[0].view(num_word, -1)

        # Compute the output
        output = F.relu(self.output_layer(hidden))
        output = F.log_softmax(output, dim=1)
        
        return output


class VectorizationTransform(object):
    __slots__ = ("vectorizer")


    def __init__(self, vectorizer):
        self.vectorizer = vectorizer


    def __call__(self, sample):
        text_array, y = sample["x"], sample["y"]

        # The text_array is only a single sample
        sequence = str(text_array)

        # Tokenize the sentence
        word_list = sequence.split()

        # Vectorization
        x = np.zeros((len(word_list), 1, self.vectorizer.vector_length), dtype=float)
        for word_index, word in enumerate(word_list):
            x[word_index][0] = self.vectorizer.vectorize_word(word)

        return { "x": torch.from_numpy(x), "y": torch.from_numpy(y) }


class RnnEmojiPredictor(NeuralEmojiPredictor):
    __slots__ = ("vectorizer", "model")


    def __init__(self, vectorizer=None, model=None, name="RnnEmojiPredictor", **kwds):
        super().__init__(name=name, **kwds)

        # For the vectorizer
        self.vectorizer = vectorizer
        if self.vectorizer is None:
            self.vectorizer = GloveVectorizer()
        
        # For the model
        self.model = model
        if self.model is None:
            self.model = Rnn(self.vectorizer.vector_length, 20, 64)


    def train(self, training_dataset, validation_dataset=None, device=None, learning_rate=1e-3, momentum=0.9, num_epoch=100):
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

        
        def collate_function(samples):
            # print("type(samples):", type(samples)) # <class 'list'>
            # print("type(samples[0]):", type(samples[0])) # <class 'dict'>
            # print("len(samples):", len(samples)) # batch_size
            batch_size = len(samples)

            # Find the maximum length among the sequences 
            maximum_length = 0
            vector_length = samples[0]["x"].shape[2]
            for sample in samples:
                x = sample["x"] # numpy.ndarray of shape (len(word_list), 1, vector_length)
                length = x.shape[0]
                if length > maximum_length:
                    maximum_length = length

            # Construct the x batch by padding with zeros
            x_batch = np.zeros((maximum_length, batch_size, vector_length), dtype=float)
            for sample_index, sample in enumerate(samples):
                x = sample["x"] # numpy.ndarray of shape (len(word_list), 1, vector_length)
                length = x.shape[0]
                for word_index in range(length):
                    x_batch[word_index][sample_index] = x[word_index][0]
            
            # Construct the y batch
            y_batch = np.stack([ sample["y"] for sample in samples ])

            return { "x": torch.from_numpy(x_batch), "y": torch.from_numpy(y_batch) }


        # Create the DataLoader
        training_dataset_torch = EmojiPredictionDataset(training_dataset, transform=VectorizationTransform(self.vectorizer))
        training_data_loader = DataLoader(training_dataset_torch, batch_size=32, shuffle=False, collate_fn=collate_function)
        if validation_dataset:
            validation_dataset_torch = EmojiPredictionDataset(validation_dataset, transform=VectorizationTransform(self.vectorizer))
            validation_data_loader = DataLoader(validation_dataset_torch, batch_size=32, shuffle=False, collate_fn=collate_function)

        # Define the optimizer and loss function
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum)
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
    
    
    def validate(self, validation_dataset, device=None):
        # # If device is not set, then use GPU if available
        # if device is None:
        #     if torch.cuda.is_available():
        #         device = torch.device('cuda')
        #         print("GPU is available. Using GPU.")
        #     else:
        #         device = torch.device('cpu')
        # print("device:", device)
        # TODO: PyTorch dynamic quantization does not support CUDA for now
        device = torch.device('cpu')

        def collate_function(samples):
            # print("type(samples):", type(samples)) # <class 'list'>
            # print("type(samples[0]):", type(samples[0])) # <class 'dict'>
            # print("len(samples):", len(samples)) # batch_size
            batch_size = len(samples)

            # Find the maximum length among the sequences 
            maximum_length = 0
            vector_length = samples[0]["x"].shape[2]
            for sample in samples:
                x = sample["x"] # numpy.ndarray of shape (len(word_list), 1, vector_length)
                length = x.shape[0]
                if length > maximum_length:
                    maximum_length = length

            # Construct the x batch by padding with zeros
            x_batch = np.zeros((maximum_length, batch_size, vector_length), dtype=float)
            for sample_index, sample in enumerate(samples):
                x = sample["x"] # numpy.ndarray of shape (len(word_list), 1, vector_length)
                length = x.shape[0]
                for word_index in range(length):
                    x_batch[word_index][sample_index] = x[word_index][0]
            
            # Construct the y batch
            y_batch = np.stack([ sample["y"] for sample in samples ])

            return { "x": torch.from_numpy(x_batch), "y": torch.from_numpy(y_batch) }
        
        # Prepare the validation DataLoader
        validation_dataset_torch = EmojiPredictionDataset(validation_dataset, transform=VectorizationTransform(self.vectorizer))
        validation_data_loader = DataLoader(validation_dataset_torch, batch_size=32, shuffle=False, collate_fn=collate_function)

        self.model.to(device)
        self.model.eval()

        correct = 0
        for sample_batch in validation_data_loader:
            batch_size = sample_batch["x"].size()[1]

            # data = sample_batch["x"]
            # target = sample_batch["y"]
            data = sample_batch["x"].float()
            target = sample_batch["y"]

            # Copy data to GPU if needed
            data = data.to(device)
            target = target.to(device)

            # Inference
            # hidden = model.reset_hidden(device, batch_size)
            # for i in range(data.size()[0]):
            #     output, hidden = model(data[i], hidden)
            output = self.model(data)

            pred = output.data.max(1)[1] # Get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum()

        accuracy = correct.to(torch.float32) / len(validation_data_loader.dataset)
        
        print('Validation accuracy: {:d}/{:d} = {:.4f}%\n'.format(
            correct,
            len(validation_data_loader.dataset),
            accuracy * 100.0))


    def predict(self, text):
        # TODO: Implementation
        pass


    def unload_model(self, path_model):
        torch.save(self.model.state_dict(), path_model)
        self.path_model = path_model
        self.model = None

    
    def load_model(self):
        if self.model is None:
            self.model = Rnn(self.vectorizer.vector_length, 20, 64)
        print(self.model)
        self.model.load_state_dict(torch.load(self.path_model))


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
            batch_size = sample_batch["x"].size()[1]

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
            # hidden = model.reset_hidden(device, batch_size)
            # for i in range(data.size()[0]):
            #     output, hidden = model(data[i], hidden)
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
            batch_size = sample_batch["x"].size()[1]

            # data = sample_batch["x"]
            # target = sample_batch["y"]
            data = sample_batch["x"].float()
            target = sample_batch["y"]

            # Copy data to GPU if needed
            data = data.to(device)
            target = target.to(device)

            # Inference
            # hidden = model.reset_hidden(device, batch_size)
            # for i in range(data.size()[0]):
            #     output, hidden = model(data[i], hidden)
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
