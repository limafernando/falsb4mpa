from sklearn.linear_model import SGDClassifier

class UnfairLogisticRegression:
    def __init__(self, epochs, lr):
        self.epochs = epochs
        self.lr = lr
        self.model = SGDClassifier(random_state=0, max_iter=self.epochs, learning_rate="constant", eta0=self.lr)
    
    def fit(self, X, y):
        self.model = self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
