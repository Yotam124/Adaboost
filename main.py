import numpy as np
import pandas as pd

from Adaboost import Adaboost

if __name__ == '__main__':
    hc_dataset = pd.read_csv('HC_Body_Temperature.txt', sep="\s+", header=None, names=["x", "label", "y"], dtype=float)
    for index in range(0, len(hc_dataset)):
        if hc_dataset['label'][index] > 1:
            hc_dataset['label'][index] = -1

    # print(hc_dataset)

    adaboost = Adaboost()

    adaboost.fit(np.array(hc_dataset['x']), np.array(hc_dataset['y']), np.array(hc_dataset['label']))

    print(len(adaboost.classifiers))
