import numpy as np


class Classifier:
    def __init__(self):
        self.direction_label = 1
        self.feature_column = None
        self.threshold = None
        self.alpha = None

    def predict(self, features):
        points_size = features.shape[0]
        current_column = features[:, self.feature_column]
        predictions = np.ones(points_size)

        if self.direction_label == 1:
            predictions[current_column < self.threshold] = -1
        else:
            predictions[current_column > self.threshold] = -1

        return predictions


class Adaboost:

    def __init__(self, n_clf=8):
        self.classifiers = []
        self.n_clf = n_clf

    def fit(self, features, labels):
        points_size, features_size = labels.shape

        weights = np.full(points_size, 1 / points_size)

        self.classifiers = []

        for _ in range(self.n_clf):
            clf = Classifier()

            min_err = float('inf')
            for feature_i in range(features_size):
                features_column = features[:, feature_i]
                thresholds = np.unique(features_column)

                for threshold in thresholds:
                    p = 1
                    predictions = np.ones(points_size)
                    predictions[features_column < threshold] = -1

                    misclassified = weights[labels != predictions]
                    err = sum(misclassified)

                    if err > 0.5:
                        err = 1 - err
                        p = -1

                    if err < min_err:
                        clf.direction_label = p
                        clf.threshold = threshold
                        clf.feature_column = feature_i
                        min_err = err

                EPS = 1e-10
                clf.alpha = 0.5 * np.log((1.0 - min_err + EPS) / (min_err + EPS))

                predictions = clf.predict(features)

                weights *= np.exp(-clf.alpha * labels * predictions)

                weights /= np.sum(weights)

                self.classifiers.append(clf)

        # def predict(self, features):
        #     clf_preds = [clf.alpha * clf.predict(features) for clf in self.classifiers]
        #     labels_pred = np.sum(clf_preds)
        #     labels_pred = np.sign(labels_pred)
        #
        #     return labels_pred