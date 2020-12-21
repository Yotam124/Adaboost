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
        if x2-x1 == 0:
            return 0
        return (x - x1) * ((y2 - y1) / (x2 - x1)) + y1

    return linear_threshold


class Adaboost:

    def __init__(self, n_clf=8):
        self.classifiers = []
        self.n_clf = n_clf

    def fit(self, features_x, features_y, labels):
        points_size = labels.shape[0]

        weights = np.full(points_size, 1 / points_size)

        self.classifiers = []

        for pointA in range(points_size):
            clf = Classifier()
            min_err = float('inf')

            for pointB in range(points_size):
                if pointA == pointB:
                    continue

                threshold = linear_equation(features_x[pointA], features_y[pointA], features_x[pointB],
                                                 features_y[pointB])

                p = 1
                predictions = np.ones(points_size)
                predictions[features_y < threshold(features_x)] = -1

                misclassified = weights[labels != predictions]
                err = sum(misclassified)

                if err > 0.5:
                    err = 1 - err
                    p = -1

                if err < min_err:
                    clf.direction_label = p
                    clf.threshold = threshold
                    min_err = err

                EPS = 1e-10
                clf.alpha = 0.5 * np.log((1.0 - min_err + EPS) / (min_err + EPS))

                predictions = clf.predict(features_x, features_y)

                weights *= np.exp(-clf.alpha * labels * predictions)

                weights /= np.sum(weights)

                self.classifiers.append(clf)



    # def predict(self, features):
    #     clf_preds = [clf.alpha * clf.predict(features) for clf in self.classifiers]
    #     labels_pred = np.sum(clf_preds)
    #     labels_pred = np.sign(labels_pred)
    #
    #     return labels_pred
