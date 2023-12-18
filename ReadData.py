import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

class ReadData(object):
    def __init__(self, data, features, y):
        enc = OrdinalEncoder()
        enc.fit(data)


        self.data = data.dropna()
        self.features = features
        self.y = self.data[y]
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []


    def get_data(self):
        print(self.data)
        #print('satisfied: ' + str(len(self.y[self.y['satisfaction']=='satisfied'])/1200))
        #print('Long term: ' + str(len(self.y[self.y['satisfaction']=='neutral or dissatisfied'])/1200))
        return self.data[self.features]


    def get_y(self):
        return self.y

    def set_experiment_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.get_data(), self.get_y(), test_size=0.3)

