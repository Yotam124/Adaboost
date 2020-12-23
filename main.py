import numpy as np
import pandas as pd
import math

from Adaboost import Adaboost
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    # Load hc_dataset dataset
    hc_dataset = pd.read_csv('HC_Body_Temperature.txt', sep="\s+", header=None, names=["x", "label", "y"], dtype=float)
    hc_dataset.loc[hc_dataset['label'] > 1, 'label'] = -1

    # Load iris dataset
    iris_dataset = pd.read_csv('iris.data', sep=",", header=None, names=["col1", "x", "y", "col4", "label"])
    iris_dataset = iris_dataset[iris_dataset['label'] != 'Iris-setosa']
    del iris_dataset['col1'], iris_dataset['col4']
    iris_dataset.loc[iris_dataset['label'] == 'Iris-versicolor', 'label'] = 1
    iris_dataset.loc[iris_dataset['label'] == 'Iris-virginica', 'label'] = -1

    # print(iris_dataset)

    # print(hc_dataset)

    x_train, x_test, y_train, y_test = train_test_split(hc_dataset[['x', 'y']], hc_dataset['label'], test_size=0.5)

    adaboost = Adaboost()

    adaboost.find_all_possible_lines(np.array(x_train['x']), np.array(x_train['y']))

    adaboost.fit(np.array(x_train['x']), np.array(x_train['y']), np.array(y_train))

    # a = {1, 2, 3, 4, 5, 6}
    # list = []
    # for i in range(len(a)):
    #     for j in range(i+1, len(a)):
    #         list.append(j)
    #
    # print(list)
    # print(len(list))
    #
    # print(math.comb(len(a), 2))

    print(len(adaboost.classifiers))
    # for clf in adaboost.classifiers:
    #     print(clf.alpha)
