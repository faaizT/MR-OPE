from scipy.special import softmax
import numpy as np


class PolicyWrapper:
    def __init__(self, classifier, beta):
        self.classifier = classifier
        self.beta = beta

    def predict_proba(self, X):
        return softmax(self.classifier.predict_log_proba(X) * self.beta, axis=-1)


class PolicyWrapperAlpha:
    def __init__(self, classifier, alpha, n_actions):
        self.classifier = classifier
        self.alpha = alpha
        self.n_actions = n_actions

    def predict_proba(self, X):
        probs = np.zeros((X.shape[0], self.n_actions))
        predicted_labels = self.classifier.predict(X).reshape(-1)
        probs[np.arange(X.shape[0]), predicted_labels] = 1
        probs_soft = probs * self.alpha + (1 - self.alpha) / (self.n_actions - 1) * (
            1 - probs
        )
        return probs_soft
