import numpy as np


class Classifier:
    def __init__(self):
        self.direction = 1
        self.threshold = None
        self.alpha = None

    def predict(self, features_x, features_y):
        points_size = features_x.shape[0]
        predictions = np.ones(points_size)

        if self.direction == 1:
            predictions[features_y < self.threshold(features_x)] = -1
        else:
            predictions[features_y > self.threshold(features_x)] = -1
        return predictions


def linear_equation(x1, y1, x2, y2):
    def linear_threshold(x):
        if x2 - x1 == 0:
            if (x > x1).any:
                return -np.inf
            else:
                return np.inf
        return (x - x1) * ((y2 - y1) / (x2 - x1)) + y1

    return linear_threshold


def predict_single_point(feature_x, feature_y, clf):
    point_pred = 1
    if clf.direction == 1:
        if feature_y < clf.threshold(feature_x):
            point_pred = -1
    else:
        if feature_y > clf.threshold(feature_x):
            point_pred = -1
    return point_pred


def hk_x(feature_x, feature_y, best_clfs):
    clf_preds = [clf.alpha * predict_single_point(feature_x, feature_y, clf) for clf in best_clfs]
    labels_pred = np.sum(clf_preds)
    labels_pred = np.sign(labels_pred)

    return labels_pred


class Adaboost:

    def __init__(self, n_clf=8):
        self.classifiers = []
        self.n_clf = n_clf
        self.best_clfs = []

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

        best_clfs = []

        for _ in range(self.n_clf):
            _clf = Classifier()
            min_err = float('inf')

            for clf in self.classifiers:
                clf_direction = 1
                predictions = np.ones(points_size)

                predictions[features_y < clf.threshold(features_x)] = -1

                misclassified = weights[labels != predictions]
                err = sum(misclassified)

                if err > 0.5:
                    err = 1 - err
                    clf_direction = -1

                if err < min_err:
                    min_err = err
                    _clf = clf
                    _clf.direction = clf_direction

            eps = 1e-10
            _clf.alpha = 0.5 * np.log((1.0 - min_err + eps) / (min_err + eps))

            predictions = _clf.predict(features_x, features_y)

            weights *= np.exp(-_clf.alpha * labels * predictions)

            weights /= np.sum(weights)

            best_clfs.append(_clf)

        self.best_clfs = best_clfs

    def calc_errors(self, features_x, features_y, labels):
        errors = []
        points_size = labels.shape[0]
        i_clfs = 1
        while i_clfs <= len(self.best_clfs):
            err = 0
            partial_best_clfs = self.best_clfs[0:i_clfs]
            for point_i in range(points_size):
                err += np.sum(
                    [labels[point_i] != hk_x(features_x[point_i], features_y[point_i], partial_best_clfs)])

            errors.append(err / points_size)
            i_clfs += 1
        return errors
