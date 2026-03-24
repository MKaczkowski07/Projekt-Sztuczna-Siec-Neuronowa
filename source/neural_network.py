import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size=1, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.cache = {}

        self.parameters = {
            'W1': np.random.randn(hidden_size, input_size) * np.sqrt(2. / input_size),
            'b1': np.zeros((hidden_size, 1)),
            'W2': np.random.randn(output_size, hidden_size) * np.sqrt(2. / hidden_size),
            'b2': np.zeros((output_size, 1))
        }

    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        return (Z > 0).astype(float)

    def forward_propagation(self, X):
        W1, b1 = self.parameters['W1'], self.parameters['b1']
        W2, b2 = self.parameters['W2'], self.parameters['b2']

        Z1 = np.dot(W1, X) + b1
        A1 = self.relu(Z1)

        Z2 = np.dot(W2, A1) + b2
        A2 = Z2

        self.cache = {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2}
        return A2

    def compute_loss(self, Y, Y_pred):
        m = Y.shape[1]
        loss = (1 / (2 * m)) * np.sum(np.square(Y_pred - Y))
        return loss

    def backward_propagation(self, X, Y):
        m = X.shape[1]
        A1, A2, Z1 = self.cache['A1'], self.cache['A2'], self.cache['Z1']
        W2 = self.parameters['W2']

        dZ2 = A2 - Y
        dW2 = (1 / m) * np.dot(dZ2, A1.T)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

        dZ1 = np.dot(W2.T, dZ2) * self.relu_derivative(Z1)
        dW1 = (1 / m) * np.dot(dZ1, X.T)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

        gradients = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
        return gradients

    def update_parameters(self, gradients):
        self.parameters['W1'] -= self.learning_rate * gradients['dW1']
        self.parameters['b1'] -= self.learning_rate * gradients['db1']
        self.parameters['W2'] -= self.learning_rate * gradients['dW2']
        self.parameters['b2'] -= self.learning_rate * gradients['db2']

    def train(self, X_train, Y_train, epochs, print_cost=True):
        for i in range(epochs):
            Y_pred = self.forward_propagation(X_train)

            loss = self.compute_loss(Y_train, Y_pred)

            gradients = self.backward_propagation(X_train, Y_train)

            self.update_parameters(gradients)

            if print_cost and i % 100 == 0:
                print(f"Epoka {i}, Błąd (MSE): {loss:.4f}")

    def predict(self, X):
        return self.forward_propagation(X)