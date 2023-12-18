import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import cross_val_score

import time

class DecisionTree(object):
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.best_alpha = 0
        self.ccp_alphas = []
        self.best_score = 0
        self.model = None
        self.fit_time = 0
        self.train_score = 0
        self.train_time = 0
        self.test_score = 0
        self.test_time = 0

        dt = DecisionTreeClassifier()

        pruning = dt.cost_complexity_pruning_path(self.X_train, self.y_train)
        alphas = pruning.ccp_alphas
        self.ccp_alphas = alphas
        self.best_depth = 0
        self.model_depth = None

    def setup_dt(self):
        self.model = DecisionTreeClassifier(random_state=0, ccp_alpha=self.best_alpha)
        self.model_depth = DecisionTreeClassifier(random_state=0, max_depth=self.best_depth)


    def learning(self, location):
        train_size, train_scores, test_scores = learning_curve(self.model, self.X_train, self.y_train, train_sizes=np.linspace(0.1, 0.9, 9), scoring='recall')

        train_mean = np.mean(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        plt.plot(train_size, train_mean, label='Train')
        plt.plot(train_size, test_mean, label='Cross Validation')
        plt.title('Decision Tree Learning Curve')
        plt.xlabel('Sample Size')
        plt.ylabel('Recall')
        plt.legend()
        plt.savefig(location + '/DecisionTree_Learning_CCP.png')
        plt.clf()

        train_size, train_scores, test_scores = learning_curve(self.model_depth, self.X_train, self.y_train,
                                                               train_sizes=np.linspace(0.1, 0.9, 9), scoring='recall')

        train_mean = np.mean(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        plt.plot(train_size, train_mean, label='Train')
        plt.plot(train_size, test_mean, label='Cross Validation')
        plt.title('Decision Tree Learning Curve')
        plt.xlabel('Sample Size')
        plt.ylabel('Recall')
        plt.legend()
        plt.savefig(location + '/DecisionTree_Learning_depth.png')
        plt.clf()

    def validation(self, location):
        train_scores, test_scores = validation_curve(DecisionTreeClassifier(), self.X_train, self.y_train, param_name='ccp_alpha', param_range=self.ccp_alphas, cv=5, scoring = 'recall')

        train_means = np.mean(train_scores, axis=1)
        test_means = np.mean(test_scores, axis=1)

        alpha_idx = np.argmax(test_means)
        self.best_alpha = self.ccp_alphas[alpha_idx]
        print('best alpha: ' + str(self.best_alpha))
        print('test mean: ' + str(test_means[alpha_idx]))
        plt.plot(self.ccp_alphas, train_means, label='Train')
        plt.plot(self.ccp_alphas, test_means, label='Cross Validation')
        plt.title('Decision Tree Validation Curve')
        plt.xlabel('Complexity Parameter')
        plt.ylabel('Recall')
        plt.legend()
        plt.savefig(location + '/DecisionTree_Validation_CCP.png')
        plt.clf()


        max_depth = [2, 4, 6, 8, 10, 12]
        train_scores, test_scores = validation_curve(DecisionTreeClassifier(), self.X_train, self.y_train,
                                                     param_name='max_depth', param_range=max_depth, cv=5,
                                                     scoring='recall')

        train_means = np.mean(train_scores, axis=1)
        test_means = np.mean(test_scores, axis=1)

        depth_idx = np.argmax(test_means)
        self.best_depth = max_depth[depth_idx]
        print('best depth: ' + str(self.best_depth))
        print('test mean: ' + str(test_means[depth_idx]))
        plt.plot(max_depth, train_means, label='Train')
        plt.plot(max_depth, test_means, label='Cross Validation')
        plt.title('Decision Tree Validation Curve')
        plt.xlabel('Max Depth')
        plt.ylabel('Recall')
        plt.legend()
        plt.savefig(location + '/DecisionTree_Validation_max_depth.png')
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
        print('Decision Tree Fit Time: ' + str(self.fit_time))
        print('Decision Tree Train Time: ' + str(self.train_time))
        print('Decision Tree Test Time: ' + str(self.test_time))
        print('Decision Tree Train Score: ' + str(self.train_score))
        print('Decision Tree Test Score: ' + str(self.test_score))

    def test_depth(self):
        fit_start = time.time()
        self.model_depth.fit(self.X_train, self.y_train)
        fit_stop = time.time()

        train_start = time.time()
        train_predictions = self.model_depth.predict(self.X_train)
        train_stop = time.time()

        test_start = time.time()
        test_predictions = self.model_depth.predict(self.X_test)
        test_stop = time.time()

        self.train_score = metrics.recall_score(self.y_train, train_predictions)
        self.fit_time = fit_stop - fit_start
        self.train_time = train_stop - train_start
        self.test_score = metrics.recall_score(self.y_test, test_predictions)
        self.test_time = test_stop - test_start
        print('Decision Tree Depth Fit Time: ' + str(self.fit_time))
        print('Decision Tree Depth Train Time: ' + str(self.train_time))
        print('Decision Tree Depth Test Time: ' + str(self.test_time))
        print('Decision Tree Depth Train Score: ' + str(self.train_score))
        print('Decision Tree Depth Test Score: ' + str(self.test_score))

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
        print('Decision Tree Fit Time: ' + str(self.fit_time))
        print('Decision Tree Train Time: ' + str(self.train_time))
        print('Decision Tree Test Time: ' + str(self.test_time))
        print('Decision Tree Train Score: ' + str(self.train_score))
        print('Decision Tree Test Score: ' + str(self.test_score))

    def test_depth_2(self):
        fit_start = time.time()
        self.model_depth.fit(self.X_train, self.y_train)
        fit_stop = time.time()

        train_start = time.time()
        train_predictions = self.model_depth.predict(self.X_train)
        train_stop = time.time()

        test_start = time.time()
        test_predictions = self.model_depth.predict(self.X_test)
        test_stop = time.time()

        self.train_score = metrics.recall_score(self.y_train, train_predictions,average='micro')
        self.fit_time = fit_stop - fit_start
        self.train_time = train_stop - train_start
        self.test_score = metrics.recall_score(self.y_test, test_predictions,average='micro')
        self.test_time = test_stop - test_start
        print('Decision Tree Fit Time: ' + str(self.fit_time))
        print('Decision Tree Train Time: ' + str(self.train_time))
        print('Decision Tree Test Time: ' + str(self.test_time))
        print('Decision Tree Train Score: ' + str(self.train_score))
        print('Decision Tree Test Score: ' + str(self.test_score))



