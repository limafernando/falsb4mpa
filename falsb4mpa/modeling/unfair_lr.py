from sklearn.linear_model import SGDClassifier

class UnfairLogisticRegression:
    def __init__(self, epochs, lr):
        self.clf = SGDClassifier(random_state=0, max_iter=epochs, learning_rate="constant", eta0=lr)
    
    def fit(self, X, y):
        self.clf = self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)
