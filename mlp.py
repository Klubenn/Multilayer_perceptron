import argparse
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Neuron:
    def __init__(self, weight=None):
        self.weight = weight if weight is not None else random.random()
        pass

    # def


class Model:
    def __init__(self, input_units: int, output_units: int, hidden_units: int, hidden_layers: int):
        pass

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