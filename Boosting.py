import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
import time
from sklearn.ensemble import AdaBoostClassifier

class Boosting(object):
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.best_n_estimators = 0
        self.n_estimators = []
        self.best_score = 0
        self.model = None
        self.fit_time = 0
        self.train_score = 0
        self.train_time = 0
        self.test_score = 0
        self.test_time = 0
        self.learning_rate = []
        self.best_learning_rate = 0
        self.model_learning_rate = None


    def setup(self):
        # Create adaboost classifier object
        self.model = AdaBoostClassifier(n_estimators=self.best_n_estimators)
        self.model_learning_rate = AdaBoostClassifier(learning_rate=self.best_learning_rate)

    def learning(self, location):
        train_size, train_scores, test_scores = learning_curve(self.model, self.X_train, self.y_train,
                                                               train_sizes=np.linspace(0.1, 0.9, 9), scoring='recall')

        train_means = np.mean(train_scores, axis=1)
        test_means = np.mean(test_scores, axis=1)
        plt.plot(train_size, train_means, label='Train')
        plt.plot(train_size, test_means, label='Cross Validation')
        plt.title('Adaboost Learning Curve')
        plt.xlabel('Sample Size')
        plt.ylabel('Recall')
        plt.legend()
        plt.savefig(location + '/Adaboost_Learning.png')
        plt.clf()

    def validation(self, location):
        self.n_estimators = range(1,100)

        train_scores, test_scores = validation_curve(AdaBoostClassifier(), self.X_train, self.y_train,
                                                     param_name='n_estimators', param_range=self.n_estimators,
                                                     cv=5, scoring='recall')

        train_means = np.mean(train_scores, axis=1)
        test_means = np.mean(test_scores, axis=1)

        n_estimators_idx = np.argmax(test_means)
        self.best_n_estimators = self.n_estimators[n_estimators_idx]
        print("Best N Estimators: " + str(self.best_n_estimators))
        print("test mean: " + str(test_means[n_estimators_idx]))

        plt.plot(self.n_estimators, train_means, label='Train')
        plt.plot(self.n_estimators, test_means, label='Cross Validation')
        plt.title('Adaboost Validation Curve')
        plt.xlabel('Number of Estimators')
        plt.ylabel('Recall')
        plt.legend()
        plt.savefig(location + '/Adaboost_Validation_Estimators.png')
        plt.clf()

        self.learning_rate = np.linspace(1.0, 10.0, 9)

        train_scores, test_scores = validation_curve(AdaBoostClassifier(), self.X_train, self.y_train,
                                                     param_name='learning_rate', param_range=self.learning_rate,
                                                     cv=5, scoring='recall')

        train_means = np.mean(train_scores, axis=1)
        test_means = np.mean(test_scores, axis=1)

        learning_rate_idx = np.argmax(test_means)
        self.best_learning_rate = self.learning_rate[learning_rate_idx]
        print("Best Learning Rate: " + str(self.best_learning_rate))
        print("test mean: " + str(test_means[learning_rate_idx]))

        plt.plot(self.learning_rate, train_means, label='Train')
        plt.plot(self.learning_rate, test_means, label='Cross Validation')
        plt.title('Adaboost Validation Curve')
        plt.xlabel('Learning Rate')
        plt.ylabel('Recall')
        plt.legend()
        plt.savefig(location + '/Adaboost_Validation_Learning_Rate.png')
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
        print('Adaboost Fit Time: ' + str(self.fit_time))
        print('Adaboost Train Time: ' + str(self.train_time))
        print('Adaboost Test Time: ' + str(self.test_time))
        print('Adaboost Train Score: ' + str(self.train_score))
        print('Adaboost Test Score: ' + str(self.test_score))

        fit_start = time.time()
        self.model_learning_rate.fit(self.X_train, self.y_train)
        fit_stop = time.time()

        train_start = time.time()
        train_predictions = self.model_learning_rate.predict(self.X_train)
        train_stop = time.time()

        test_start = time.time()
        test_predictions = self.model_learning_rate.predict(self.X_test)
        test_stop = time.time()

        self.train_score = metrics.recall_score(self.y_train, train_predictions)
        self.fit_time = fit_stop - fit_start
        self.train_time = train_stop - train_start
        self.test_score = metrics.recall_score(self.y_test, test_predictions)
        self.test_time = test_stop - test_start
        print('Adaboost Learning Rate Fit Time: ' + str(self.fit_time))
        print('Adaboost Learning Rate Train Time: ' + str(self.train_time))
        print('Adaboost Learning Rate Test Time: ' + str(self.test_time))
        print('Adaboost Learning Rate Train Score: ' + str(self.train_score))
        print('Adaboost Learning Rate Test Score: ' + str(self.test_score))

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
        print('Adaboost Fit Time: ' + str(self.fit_time))
        print('Adaboost Train Time: ' + str(self.train_time))
        print('Adaboost Test Time: ' + str(self.test_time))
        print('Adaboost Train Score: ' + str(self.train_score))
        print('Adaboost Test Score: ' + str(self.test_score))
