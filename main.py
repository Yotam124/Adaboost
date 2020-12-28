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
    iris_dataset.loc[iris_dataset['label'] == 'Iris-versicolor', 'label'] = 1.0
    iris_dataset.loc[iris_dataset['label'] == 'Iris-virginica', 'label'] = -1.0
    iris_dataset['label'] = pd.to_numeric(iris_dataset['label'])

    iterations = 100
    # ------------------------ Adaboost for hc_dataset ------------------------
    adaboost_hc = Adaboost()
    adaboost_hc.find_all_possible_lines(np.array(hc_dataset['x']), np.array(hc_dataset['y']))

    hc_errors = pd.DataFrame({'emp_err': [0, 0, 0, 0, 0, 0, 0, 0], 'true_err': [0, 0, 0, 0, 0, 0, 0, 0]})
    for _ in range(iterations):
        x_train, x_test, y_train, y_test = train_test_split(hc_dataset[['x', 'y']], hc_dataset['label'], test_size=0.5)
        adaboost_hc.fit(np.array(x_train['x']), np.array(x_train['y']), np.array(y_train))

        emp_errs = adaboost_hc.calc_errors(np.array(x_train['x']), np.array(x_train['y']), np.array(y_train))
        true_errs = adaboost_hc.calc_errors(np.array(x_test['x']), np.array(x_test['y']), np.array(y_test))

        hc_errors['emp_err'] += emp_errs
        hc_errors['true_err'] += true_errs

    hc_errors /= iterations
    print('-------------- HC dataset --------------')
    print(hc_errors)

    # ------------------------ Adaboost for iris_dataset ------------------------
    adaboost_iris = Adaboost()
    adaboost_iris.find_all_possible_lines(np.array(iris_dataset['x']), np.array(iris_dataset['y']))

    iris_errors = pd.DataFrame({'emp_err': [0, 0, 0, 0, 0, 0, 0, 0], 'true_err': [0, 0, 0, 0, 0, 0, 0, 0]})
    for _ in range(iterations):
        x_train, x_test, y_train, y_test = train_test_split(iris_dataset[['x', 'y']], iris_dataset['label'],
                                                            test_size=0.5)
        adaboost_iris.fit(np.array(x_train['x']), np.array(x_train['y']), np.array(y_train))

        emp_errs = adaboost_iris.calc_errors(np.array(x_train['x']), np.array(x_train['y']), np.array(y_train))
        true_errs = adaboost_iris.calc_errors(np.array(x_test['x']), np.array(x_test['y']), np.array(y_test))

        iris_errors['emp_err'] += emp_errs
        iris_errors['true_err'] += true_errs

    iris_errors /= iterations
    print('-------------- Iris dataset --------------')
    print(iris_errors)
