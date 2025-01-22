#!/usr/bin/env python3
"""
Defines the DeepNeuralNetwork class for multiclass classification.
"""

import numpy as np


class DeepNeuralNetwork:
    """
    Class that represents a deep neural network for multiclass classification.
    """

    def __init__(self, nx, layers):
        """
        Initialize the deep neural network.

        Parameters:
            nx (int): Number of input features.
            layers (list): List of integers representing the number of nodes in
                           each layer.

        Raises:
            TypeError: If nx is not an integer or layers is not a list of positive integers.
            ValueError: If nx is less than 1.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0 or not all(
            isinstance(l, int) and l > 0 for l in layers
        ):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i, layer_nodes in enumerate(layers):
            self.__weights[f"W{i + 1}"] = np.random.randn(
                layer_nodes, nx if i == 0 else layers[i - 1]
            ) * np.sqrt(2 / (nx if i == 0 else layers[i - 1]))
            self.__weights[f"b{i + 1}"] = np.zeros((layer_nodes, 1))

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    def forward_prop(self, X):
        """
        Perform forward propagation.

        Parameters:
            X (numpy.ndarray): Input data of shape (nx, m).

        Returns:
            A (numpy.ndarray): Output of the network.
            cache (dict): Dictionary containing all intermediate activations.
        """
        self.__cache["A0"] = X
        for l in range(1, self.L + 1):
            W = self.__weights[f"W{l}"]
            b = self.__weights[f"b{l}"]
            Z = np.matmul(W, self.__cache[f"A{l - 1}"]) + b
            if l == self.L:  # Output layer with softmax activation
                exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))  # Stability
                A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
            else:  # Hidden layers with sigmoid activation
                A = 1 / (1 + np.exp(-Z))
            self.__cache[f"A{l}"] = A
        return A, self.__cache

    def cost(self, Y, A):
        """
        Calculate the cost using categorical cross-entropy.

        Parameters:
            Y (numpy.ndarray): One-hot labels of shape (classes, m).
            A (numpy.ndarray): Predictions of shape (classes, m).

        Returns:
            float: Cost value.
        """
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A)) / m
        return cost

    def evaluate(self, X, Y):
        """
        Evaluate the network's predictions.

        Parameters:
            X (numpy.ndarray): Input data of shape (nx, m).
            Y (numpy.ndarray): One-hot labels of shape (classes, m).

        Returns:
            (numpy.ndarray, float): Predicted labels and cost.
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        predictions = np.argmax(A, axis=0)
        return predictions, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Perform one pass of gradient descent on the network.

        Parameters:
            Y (numpy.ndarray): One-hot labels of shape (classes, m).
            cache (dict): Dictionary of cached values from forward propagation.
            alpha (float): Learning rate.
        """
        m = Y.shape[1]
        L = self.L
        dz = cache[f"A{L}"] - Y
        for l in range(L, 0, -1):
            A_prev = cache[f"A{l - 1}"]
            W = self.__weights[f"W{l}"]
            dW = np.matmul(dz, A_prev.T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m
            if l > 1:
                dz = np.matmul(W.T, dz) * (A_prev * (1 - A_prev))
            self.__weights[f"W{l}"] -= alpha * dW
            self.__weights[f"b{l}"] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """
        Train the neural network.

        Parameters:
            X (numpy.ndarray): Input data of shape (nx, m).
            Y (numpy.ndarray): One-hot labels of shape (classes, m).
            iterations (int): Number of iterations.
            alpha (float): Learning rate.
            verbose (bool): Whether to print progress.
            graph (bool): Whether to plot cost.
            step (int): Step interval for verbose and graph.

        Returns:
            (numpy.ndarray, float): Predicted labels and cost after training.
        """
        if not isinstance(iterations, int) or iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float) or alpha <= 0:
            raise ValueError("alpha must be a positive float")
        if verbose or graph:
            if not isinstance(step, int) or step <= 0 or step > iterations:
                raise ValueError("step must be a positive integer and <= iterations")

        costs = []
        for i in range(iterations):
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)
            if (verbose or graph) and (i % step == 0 or i == iterations - 1):
                cost = self.cost(Y, A)
                if verbose:
                    print(f"Cost after {i} iterations: {cost}")
                if graph:
                    costs.append((i, cost))

        if graph:
            import matplotlib.pyplot as plt
            x, y = zip(*costs)
            plt.plot(x, y)
            plt.xlabel("Iterations")
            plt.ylabel("Cost")
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """
        Save the instance to a file in pickle format.
        """
        import pickle
        if not filename.endswith(".pkl"):
            filename += ".pkl"
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """
        Load a pickled instance of DeepNeuralNetwork.

        Returns:
            DeepNeuralNetwork or None.
        """
        import pickle
        try:
            with open(filename, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None
