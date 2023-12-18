from sklearn import svm
import numpy as np
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
import time

class SupportVectorMachine(object):
    def __init__(self, X_train, X_test, y_train, y_test, kernel):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.kernel = kernel
        self.best_c = 0
        self.c = []
        self.best_score = 0
        self.model = None
        self.fit_time = 0
        self.train_score = 0
        self.train_time = 0
        self.test_score = 0
        self.test_time = 0

    def setup(self):
        self.model = svm.SVC(kernel=self.kernel, C=self.best_c)

    def learning(self, location):
        train_size, train_scores, test_scores = learning_curve(self.model, self.X_train, self.y_train,
                                                               train_sizes=np.linspace(0.1, 0.9, 9), scoring='recall')

        train_means = np.mean(train_scores, axis=1)
        test_means = np.mean(test_scores, axis=1)
        plt.plot(train_size, train_means, label='Train')
        plt.plot(train_size, test_means, label='Cross Validation')
        plt.title('SVM Learning Curve')
        plt.xlabel('Sample Size')
        plt.ylabel('Recall')
        plt.legend()
        plt.savefig(location + '/SVM_Learning-' + self.kernel + '.png')
        plt.clf()

    def validation(self, location):
        self.c = np.linspace(0.1, 0.9, 9)

        train_scores, test_scores = validation_curve(svm.SVC(), self.X_train, self.y_train,
                                                     param_name='C', param_range=self.c,
                                                     cv=5, scoring='recall')

        train_means = np.mean(train_scores, axis=1)
        test_means = np.mean(test_scores, axis=1)

        c_idx = np.argmax(test_means)
        self.best_c = self.c[c_idx]
        print('best c: ' + str(self.best_c))
        print('best c: ' + str(test_means[c_idx]))

        plt.plot(self.c, train_means, label='Train')
        plt.plot(self.c, test_means, label='Cross Validation')
        plt.title('SVM Validation Curve')
        plt.xlabel('C Value')
        plt.ylabel('Recall')
        plt.legend()
        plt.savefig(location + '/SVM_Validation-' + self.kernel + '.png')
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
        print('SVM Fit Time: ' + str(self.fit_time))
        print('SVM Train Time: ' + str(self.train_time))
        print('SVM Test Time: ' + str(self.test_time))
        print('SVM Train Score: ' + str(self.train_score))
        print('SVM Test Score: ' + str(self.test_score))

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
        self.test_score = metrics.recall_score(self.y_test, test_predictions,average = 'micro')
        self.test_time = test_stop - test_start
        print('SVM Fit Time: ' + str(self.fit_time))
        print('SVM Train Time: ' + str(self.train_time))
        print('SVM Test Time: ' + str(self.test_time))
        print('SVM Train Score: ' + str(self.train_score))
        print('SVM Test Score: ' + str(self.test_score))
