import numpy as np
import pandas as pd
from DatasetParser import parse
from Adaboost import Adaboost

if __name__ == '__main__':
    hc_dataset = parse("HC_Body_Temperature.txt")
    iris_dataset = np.array(parse("iris.data"), dtype=object)
    # print(hc_dataset)
    # print(iris_dataset)

    for i in range(len(iris_dataset)):
        print(iris_dataset[i][0])


clf = Adaboost(8)

# clf.fit(iris_dataset[:, 0], iris_dataset[:, 1])
# print(clf.classifiers)
