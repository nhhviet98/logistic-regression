import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class MyLogisticRegression():
    def __init__(self, max_iter=500, lr=0.00001):
        self.max_iter = max_iter
        self.lr = lr
        self.m = None
        self.n = None
        self.w = None
        self.num_class = None
        self.y_train_one_hot = None

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        '''
        sigmoid function
        :param z:
        :return: np.ndarray
        '''
        return 1/(1 + np.exp(-z))

    @staticmethod
    def one_hot(y: list, nb_class: int) -> np.ndarray:
        '''
        One hot vectors for labels
        :param y: list
            List of labels
        :param nb_class: int
            Number of class in data set
        :return:
        '''
        y = np.array(y)
        y_one_hot = np.eye(nb_class)[y]
        return y_one_hot

    @staticmethod
    def accuracy_score(y_test: list, y_pred: list) -> float:
        '''
        Calculate accuracy of model
        :param y_test: list
            Labels of test set
        :param y_pred:
            Predicted labels of model
        :return: float
            Accuracy of model
        '''
        return sum(np.equal(y_test, y_pred)) / len(y_test)

    @staticmethod
    def plot_loss(total_lost: list):
        '''
        Plot loss of training
        :param total_lost: list of history lost during training
        :return: None
        '''
        plt.figure(figsize=(15, 6))
        plt.plot(range(len(total_lost)), total_lost)
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.show()

    def cost(self, z: np.ndarray) -> float:
        '''
        Calculate loss of each epochs
        :param z: np.ndarray
            z of model in training phase
        :return: float
            Loss of each epochs
        '''
        cost = np.sum(-1/self.m*(self.y_train_one_hot*np.log(z) + (1-self.y_train_one_hot)*np.log(1-z)))/self.num_class
        return cost

    def fit(self, x_train: np.ndarray, y_train: list) -> list:
        '''
        Fit data to training
        :param x_train: np.ndarray
            Input of training set
        :param y_train: list
            Output of training set
        :return: list
            History of loss during training
        '''
        self.m = x_train.shape[0]
        self.n = x_train.shape[1]
        self.num_class = np.max(y_train) + 1
        self.w = np.zeros((self.num_class, self.n))
        cost = np.zeros(self.max_iter)
        self.y_train_one_hot = self.one_hot(y_train, self.num_class)
        total_lost = []
        for i in tqdm(range(self.max_iter)):
            z = self.sigmoid(x_train @ self.w.T)
            cost = self.cost(z)
            self.w = self.w + self.lr*(self.y_train_one_hot - z).T @ x_train
            total_lost.append(cost)
        return total_lost

    def predict(self, x_test: np.ndarray) -> list:
        '''
        Predict labels for test set
        :param x_test: np.ndarray
            Input of test set
        :return:
            Predicted labels for test set
        '''
        z = self.sigmoid(x_test @ self.w.T)
        y_pred = list(np.argmax(z, axis=1))
        print('yeah yeah')
        return y_pred
