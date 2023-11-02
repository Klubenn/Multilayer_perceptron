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
        self.input_features = input_features
        self.output_features = output_features
        self.hidden_layers = hidden_layers
        self.hidden_features = hidden_features
        self.activation_function = activation_function

        self.weights = self._create_layers(initialize=True)
        self.biases = np.empty(hidden_layers + 1, dtype='float32')
        self.neurons_outputs = self._create_layers(initialize=False)

    def _create_layers(self, initialize: bool):
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

    def calculate_neuron_input(self, layer, neuron_number):



    def forward_pass(self, data):
        pass

    def calculate_loss(self):
        pass

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