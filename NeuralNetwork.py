import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import cross_val_score
import time


class NeuralNetwork(object):
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.best_hidden_layer = (0,)
        self.hidden_layers = []
        self.best_score = 0
        self.model = None
        self.fit_time = 0
        self.train_score = 0
        self.train_time = 0
        self.test_score = 0
        self.test_time = 0

    def set_hidden_layers(self):
        # X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3)
        layers = range(1,len(self.X_train.axes[1]))

        best_hidden_layer = 0
        best_score = 0
        mean, std = [], []

        for i in layers:
            # if i > 0:
            mlp = MLPClassifier(hidden_layer_sizes=(i,), max_iter=500)
            mlp.fit(self.X_train, self.y_train)
            # dts.append(tree)
            score = cross_val_score(mlp, self.X_train, self.y_train, cv=5, scoring='recall')
            mean.append(score.mean())
            mean_score = score.mean()
            self.hidden_layers.append((i, i, i))

            if mean_score > best_score:
                best_hidden_layer = i
                best_score = mean_score

        self.best_hidden_layers = best_hidden_layer
        self.best_score = best_score

        print('hidden layer: ' + str(self.best_hidden_layer))
        print('score: ' + str(self.best_score))


    def setup(self):
        self.model = MLPClassifier(hidden_layer_sizes=self.best_hidden_layer)

    def learning(self, location):
        train_size, train_scores, test_scores = learning_curve(self.model, self.X_train, self.y_train,
                                                               train_sizes=np.linspace(0.1, 0.9, 9), scoring='recall')

        train_means = np.mean(train_scores, axis=1)
        test_means = np.mean(test_scores, axis=1)
        plt.plot(train_size, train_means, label='Train')
        plt.plot(train_size, test_means, label='Cross Validation')
        plt.title('Neural Network Learning Curve')
        plt.xlabel('Sample Size')
        plt.ylabel('Recall')
        plt.legend()
        plt.savefig(location + '/NeuralNetwork_Learning.png')
        plt.clf()

    def validation(self, location):
        for i in range(1,self.X_train.shape[1] + 1):
            self.hidden_layers.append((i,))
        print(self.hidden_layers)
        train_scores, test_scores = validation_curve(MLPClassifier(), self.X_train, self.y_train,
                                                                 param_name='hidden_layer_sizes', param_range=self.hidden_layers,
                                                                 cv=5, scoring='recall')

        train_means = np.mean(train_scores, axis=1)
        test_means = np.mean(test_scores, axis=1)
        layer_sizes = range(1,self.X_train.shape[1]+1)
        print(self.hidden_layers)
        print(train_means)

        hidden_layer_idx = np.argmax(test_means)

        self.best_hidden_layer = self.hidden_layers[hidden_layer_idx]
        print('best hidden layer: ' + str(self.best_hidden_layer))
        print('test mean: ' + str(test_means[hidden_layer_idx]))


        plt.plot(layer_sizes, train_means, label='Train')
        plt.plot(layer_sizes, test_means, label='Cross Validation')
        plt.title('Neural Network Validation Curve')
        plt.xlabel('Hidden Layers')
        plt.ylabel('Recall')
        plt.legend()
        plt.savefig(location + '/NeuralNetwork_Validation.png')
        plt.clf()

    def test_model(self):
        fit_start = time.time()
        self.model.fit(self.X_train, self.y_train)
        fit_stop = time.time()

        train_start = time.time()
        train_predictions = self.model.predict(self.X_train)
        train_stop = time.time()

        test_start = time.time()
        test_predictions = self.model.predict(self.X_test)
        test_stop = time.time()

        self.train_score = metrics.recall_score(self.y_train, train_predictions)
        self.fit_time = fit_stop - fit_start
        self.train_time = train_stop - train_start
        self.test_score = metrics.recall_score(self.y_test, test_predictions)
        self.test_time = test_stop - test_start
        print('Neural Network Fit Time: ' + str(self.fit_time))
        print('Neural Network Train Time: ' + str(self.train_time))
        print('Neural Network Test Time: ' + str(self.test_time))
        print('Neural Network Train Score: ' + str(self.train_score))
        print('Neural Network Test Score: ' + str(self.test_score))

    def test_model_2(self):
        fit_start = time.time()
        self.model.fit(self.X_train, self.y_train)
        fit_stop = time.time()

        train_start = time.time()
        train_predictions = self.model.predict(self.X_train)
        train_stop = time.time()

        test_start = time.time()
        test_predictions = self.model.predict(self.X_test)
        test_stop = time.time()

        self.train_score = metrics.recall_score(self.y_train, train_predictions,average='micro')
        self.fit_time = fit_stop - fit_start
        self.train_time = train_stop - train_start
        self.test_score = metrics.recall_score(self.y_test, test_predictions,average='micro')
        self.test_time = test_stop - test_start
        print('Neural Network Fit Time: ' + str(self.fit_time))
        print('Neural Network Train Time: ' + str(self.train_time))
        print('Neural Network Test Time: ' + str(self.test_time))
        print('Neural Network Train Score: ' + str(self.train_score))
        print('Neural Network Test Score: ' + str(self.test_score))