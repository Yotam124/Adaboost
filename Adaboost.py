import numpy as np


class Classifier:
    def __init__(self):
        self.direction_label = 1
        self.threshold = None
        self.alpha = None

    def predict(self, features_x, features_y):
        points_size = features_x.shape[0]
        predictions = np.ones(points_size)

        if self.direction_label == 1:
            predictions[features_y < self.threshold(features_x)] = -1
        else:
            predictions[features_y > self.threshold(features_x)] = -1

        return predictions


def linear_equation(x1, y1, x2, y2):
    def linear_threshold(x):
        if x2 - x1 == 0:
            return 0
        return (x - x1) * ((y2 - y1) / (x2 - x1)) + y1

    return linear_threshold


class Adaboost:

    def __init__(self, n_clf=8):
        self.classifiers = []
        self.n_clf = n_clf

    def find_all_possible_lines(self, features_x, features_y):
        points_size = features_x.shape[0]
        for pointA in range(points_size):
            for pointB in range(pointA + 1, points_size):
                clf = Classifier()
                clf.threshold = linear_equation(features_x[pointA], features_y[pointA], features_x[pointB],
                                                features_y[pointB])
                self.classifiers.append(clf)
                # print("pointA, pointB: ", "(", features_x[pointA], ",", features_y[pointA], ")", "(",
                #       features_x[pointB], ",", features_y[pointB], ")")

    def fit(self, features_x, features_y, labels):
        points_size = labels.shape[0]

        weights = np.full(points_size, 1 / points_size)

        for clf in self.classifiers:

            min_err = float('inf')

            p = 1
            predictions = np.ones(points_size)
            predictions[features_y < clf.threshold(features_x)] = -1

            misclassified = weights[labels != predictions]
            err = sum(misclassified)

            if err > 0.5:
                err = 1 - err
                p = -1

            if err < min_err:
                clf.direction_label = p
                min_err = err

            eps = 1e-10
            clf.alpha = 0.5 * np.log((1.0 - min_err + eps) / (min_err + eps))

            predictions = clf.predict(features_x, features_y)

            weights *= np.exp(-clf.alpha * labels * predictions)

            weights /= np.sum(weights)

    # def predict(self, features):
    #     clf_preds = [clf.alpha * clf.predict(features) for clf in self.classifiers]
    #     labels_pred = np.sum(clf_preds)
    #     labels_pred = np.sign(labels_pred)
    #
    #     return labels_pred
