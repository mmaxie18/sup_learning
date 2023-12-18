import pandas as pd
from DecisionTree import *
from NeuralNetwork import *
from Boosting import *
from SupportVectorMachine import *
from KNearestNeighbors import *
from ReadData import *


if __name__ == "__main__":

    # reading csv files
    data = pd.read_csv('data/mushroom/agaricus-lepiota.data', header=None)
    data_2 = pd.read_csv('data/air_travel/modified_air_sat.csv')
    features_1 = list(range(1,23))
    features_2 = list(range(1,24))

    enc = OrdinalEncoder()
    enc.fit(data)
    data = pd.DataFrame(enc.transform(data))

    enc_2 = OrdinalEncoder()
    enc_2.fit(data_2)
    data_2 = pd.DataFrame(enc_2.transform(data_2))

    mushroom_data = ReadData(data, features_1, 0)
    mushroom_data.set_experiment_data()
    print(mushroom_data.y_train)
    print(data_2)
    contra_data = ReadData(data_2, features_2, 24)
    contra_data.set_experiment_data()
    contra_data.get_data()

    # Decision Tree Implementation
    dt = DecisionTree(mushroom_data.X_train, mushroom_data.X_test, mushroom_data.y_train, mushroom_data.y_test)

    dt.validation('images_1')
    dt.setup_dt()
    dt.learning('images_1')
    dt.test_model()
    dt.test_depth()

    # Neural Network
    nn = NeuralNetwork(mushroom_data.X_train, mushroom_data.X_test, mushroom_data.y_train, mushroom_data.y_test)
    nn.validation('images_1')
    nn.setup()
    nn.learning('images_1')
    nn.test_model()

    # Boosting
    boost = Boosting(mushroom_data.X_train, mushroom_data.X_test, mushroom_data.y_train, mushroom_data.y_test)
    boost.validation('images_1')
    boost.setup()
    boost.learning('images_1')
    boost.test_model()


    # Support Vector Machines
    svm = SupportVectorMachine(mushroom_data.X_train, mushroom_data.X_test, mushroom_data.y_train, mushroom_data.y_test, 'linear')
    svm.validation('images_1')
    svm.setup()
    svm.learning('images_1')
    svm.test_model()

    svm_poly = SupportVectorMachine(mushroom_data.X_train, mushroom_data.X_test, mushroom_data.y_train, mushroom_data.y_test, 'rbf')
    svm_poly.validation('images_1')
    svm_poly.setup()
    svm_poly.learning('images_1')
    svm_poly.test_model()

    # k-Nearest Neighbors
    knn = KNearestNeighbors(mushroom_data.X_train, mushroom_data.X_test, mushroom_data.y_train, mushroom_data.y_test)
    knn.validation('images_1')
    knn.setup()
    knn.learning('images_1')
    knn.test_model()
    knn.test_distance()


    # Decision Tree Implementation
    dt = DecisionTree(contra_data.X_train, contra_data.X_test, contra_data.y_train, contra_data.y_test)
    dt.validation('images_2')
    dt.setup_dt()
    dt.learning('images_2')
    dt.test_model()
    dt.test_depth()

    # Neural Network
    nn = NeuralNetwork(contra_data.X_train, contra_data.X_test, contra_data.y_train, contra_data.y_test)
    nn.validation('images_2')
    nn.setup()
    nn.learning('images_2')
    nn.test_model()

    # Boosting
    boost = Boosting(contra_data.X_train, contra_data.X_test, contra_data.y_train, contra_data.y_test)
    boost.validation('images_2')
    boost.setup()
    boost.learning('images_2')
    boost.test_model()

    # Support Vector Machines
    svm = SupportVectorMachine(contra_data.X_train, contra_data.X_test, contra_data.y_train, contra_data.y_test, 'linear')
    svm.validation('images_2')
    svm.setup()
    svm.learning('images_2')
    svm.test_model()

    svm = SupportVectorMachine(contra_data.X_train, contra_data.X_test, contra_data.y_train, contra_data.y_test, 'rbf')
    svm.validation('images_2')
    svm.setup()
    svm.learning('images_2')
    svm.test_model()

    # k-Nearest Neighbors
    knn = KNearestNeighbors(contra_data.X_train, contra_data.X_test, contra_data.y_train, contra_data.y_test)
    knn.validation('images_2')
    knn.setup()
    knn.learning('images_2')
    knn.test_model()
    knn.test_distance()


