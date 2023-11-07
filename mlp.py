import argparse
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def relu(x: float):
    if x > 0:
        return x
    return 0

class Neuron:
    def __init__(self, weight=None):
        self.weight = weight if weight is not None else random.random()
        pass

    # def


class Model:
    def __init__(self,
                    input_features: int,
                    output_features: int,
                    hidden_layers: int,
                    hidden_features: int,
                    activation_function,
                    ):
        """
        Creates matrices for weights, biases and neuron outputs either with
        initialized values or just with zeroes.
        :param input_features: number of input features (neurons)
        :param output_features: number of output features (neurons)
        :param hidden_layers: number of hidden layers
        :param hidden_features: number of features (neurons) in each hidden layer
        :param activation_function: activation function for neuron output
        """
        self.input_features = input_features
        self.output_features = output_features
        self.hidden_layers = hidden_layers
        self.hidden_features = hidden_features
        self.activation_function = activation_function

        self.weights = self._create_matrices(initialize=True)
        self.biases = self._create_vectors(initialize=True)
        self.neurons_outputs = self._create_vectors(initialize=False)
        self.results = self._create_vectors(initialize=False)
        self.error = []

    def _create_matrices(self, initialize: bool) -> dict:
        """
        Creates the dictionary with the keys as numbers from 0 to the number
        of layers and the values are the matrices with specific shape.
        The total amount of dict keys equals to the number of hidden
        layers plus one. The shape of the matrix for each layer is the
        following: the number of rows equals to the number of neurons to the
        right and the number of columns equals to the number of neurons to the
        left. The matrix is either initialized with zeros or with random float
        numbers depending on the value of the argument.
        :param initialize: bool
        :return: dict
        """
        layers = {}
        for i in range(self.hidden_layers + 1):
            # number of matrix rows is equal to the number of input neurons (to the right)
            rows = self.hidden_features if i != self.hidden_layers else self.output_features
            # number of matrix columns is equal to the number of output neurons (to the left)
            columns = self.input_features if i == 0 else self.hidden_features
            if initialize:
                layers[i] = np.empty((rows, columns), dtype='float32')
            else:
                layers[i] = np.zeros((rows, columns), dtype='float32')
        return layers

    def _create_vectors(self, initialize: bool) -> dict:
        """
        Creates the dictionary with the keys as numbers from 0 to the number
        of layers and the values are the vectors. The total amount of dict keys
        equals to the number of hidden layers plus one. The length of the
        vector for each layer equals to the number of neurons of the current
        layer. Input layer is layer 0 and the last layer is the final hidden
        layer. The vector is either initialized with zeros or with random float
        numbers depending on the value of the argument.
        :param initialize: bool
        :return: dict
        """
        layers = {}
        for i in range(self.hidden_layers + 1):
            length = self.input_features if i == 0 else self.hidden_features
            if initialize:
                layers[i] = np.empty(length, dtype='float32')
            else:
                layers[i] = np.zeros(length, dtype='float32')
        return layers

    def calculate_perceptrons_in_layer(self, layer, neuron_number):

        pass


    def forward_pass(self, data):
        pass

    def calculate_error(self, labels):
        error = np.count_nonzero(self.results == labels)
        self.error.append(error)

    def backpropagation(self, error):
        pass


def plot(X: pd.DataFrame, labels):
    plt.scatter(x=X.loc[:, 2],
                y=X.loc[:, 3],
                c=labels)
    pass

def main():
    parser = argparse.ArgumentParser(
        prog='mlp',
        description='Calculates the probability of the cancer',
    )
    df = pd.read_csv('data.csv', index_col=0, header=None)
    df = df.replace('M', 1)
    df = df.replace('B', 0)
    print(df)

    X = df.loc[:, 2:]
    labels = df.loc[:, 1]

    print(len(X), len(labels))
    print(X.shape, labels.shape, df.shape)
    print(X.loc[:, 2])
    plot(X, labels)



if __name__ == '__main__':
    main()