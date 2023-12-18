import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
import time

class KNearestNeighbors(object):
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.best_k = 5
        self.k = []
        self.best_score = 0
        self.model = None
        self.fit_time = 0
        self.train_score = 0
        self.train_time = 0
        self.test_score = 0
        self.test_time = 0
        self.distance_model = None

    def setup(self):
        self.model = KNeighborsClassifier(n_neighbors=self.best_k)
        self.distance_model = KNeighborsClassifier(n_neighbors=self.best_k, weights='distance')

    def learning(self, location):
        train_size, train_scores, test_scores = learning_curve(self.model, self.X_train, self.y_train,
                                                               train_sizes=np.linspace(0.1, 0.9, 9), scoring='recall')

        train_means = np.mean(train_scores, axis=1)
        test_means = np.mean(test_scores, axis=1)
        plt.plot(train_size, train_means, label='Train')
        plt.plot(train_size, test_means, label='Cross Validation')
        plt.title('KNN Learning Curve')
        plt.xlabel('Sample Size')
        plt.ylabel('Recall')
        plt.legend()
        plt.savefig(location + '/KNN_Learning.png')
        plt.clf()


        train_size, train_scores, test_scores = learning_curve(self.distance_model, self.X_train, self.y_train,
                                                               train_sizes=np.linspace(0.1, 0.9, 9), scoring='recall')

        train_means = np.mean(train_scores, axis=1)
        test_means = np.mean(test_scores, axis=1)
        plt.plot(train_size, train_means, label='Train')
        plt.plot(train_size, test_means, label='Cross Validation')
        plt.title('KNN Learning Curve')
        plt.xlabel('Sample Size')
        plt.ylabel('Recall')
        plt.legend()
        plt.savefig(location + '/KNN_Learning_Distance.png')
        plt.clf()

    def validation(self, location):
        self.k = range(1,15)

        train_scores, test_scores = validation_curve(KNeighborsClassifier(), self.X_train, self.y_train,
                                                     param_name='n_neighbors', param_range=self.k,
                                                     cv=5, scoring='recall')

        train_means = np.mean(train_scores, axis=1)
        test_means = np.mean(test_scores, axis=1)

        k_idx = np.argmax(test_means)
        self.best_k = self.k[k_idx]
        print('best k: ' + str(self.best_k))
        print('best mean: ' + str(test_means[k_idx]))
        plt.plot(self.k, train_means, label='Train')
        plt.plot(self.k, test_means, label='Cross Validation')
        plt.title('KNN Validation Curve')
        plt.xlabel('K Value')
        plt.ylabel('Recall')
        plt.legend()
        plt.savefig(location + '/KNN_Validation.png')
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
        print('KNN Fit Time: ' + str(self.fit_time))
        print('KNN Train Time: ' + str(self.train_time))
        print('KNN Test Time: ' + str(self.test_time))
        print('KNN Train Score: ' + str(self.train_score))
        print('KNN Test Score: ' + str(self.test_score))

    def test_distance(self):
        fit_start = time.time()
        self.distance_model.fit(self.X_train, self.y_train)
        fit_stop = time.time()

        train_start = time.time()
        train_predictions = self.distance_model.predict(self.X_train)
        train_stop = time.time()

        test_start = time.time()
        test_predictions = self.distance_model.predict(self.X_test)
        test_stop = time.time()

        self.train_score = metrics.recall_score(self.y_train, train_predictions)
        self.fit_time = fit_stop - fit_start
        self.train_time = train_stop - train_start
        self.test_score = metrics.recall_score(self.y_test, test_predictions)
        self.test_time = test_stop - test_start
        print('KNN Fit Time: ' + str(self.fit_time))
        print('KNN Train Time: ' + str(self.train_time))
        print('KNN Test Time: ' + str(self.test_time))
        print('KNN Train Score: ' + str(self.train_score))
        print('KNN Test Score: ' + str(self.test_score))

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
        print('KNN Fit Time: ' + str(self.fit_time))
        print('KNN Train Time: ' + str(self.train_time))
        print('KNN Test Time: ' + str(self.test_time))
        print('KNN Train Score: ' + str(self.train_score))
        print('KNN Test Score: ' + str(self.test_score))

    def test_distance_2(self):
        fit_start = time.time()
        self.distance_model.fit(self.X_train, self.y_train)
        fit_stop = time.time()

        train_start = time.time()
        train_predictions = self.distance_model.predict(self.X_train)
        train_stop = time.time()

        test_start = time.time()
        test_predictions = self.distance_model.predict(self.X_test)
        test_stop = time.time()

        self.train_score = metrics.recall_score(self.y_train, train_predictions,average='micro')
        self.fit_time = fit_stop - fit_start
        self.train_time = train_stop - train_start
        self.test_score = metrics.recall_score(self.y_test, test_predictions,average='micro')
        self.test_time = test_stop - test_start
        print('KNN Fit Time: ' + str(self.fit_time))
        print('KNN Train Time: ' + str(self.train_time))
        print('KNN Test Time: ' + str(self.test_time))
        print('KNN Train Score: ' + str(self.train_score))
        print('KNN Test Score: ' + str(self.test_score))