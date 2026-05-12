from sklearn.linear_model import SGDClassifier


class VanillaLogisticRegression:
    def __init__(self, epochs, lr):
        self.epochs = epochs
        self.lr = lr
        self.model = SGDClassifier(
            loss="log_loss",  # logistic regression
            max_iter=self.epochs,
            learning_rate="constant",
            eta0=self.lr,
            shuffle=False,
        )

    def fit(self, X, y):
        self.model = self.model.fit(X, y)

    def partial_fit(self, X, y, classes=[0, 1]):
        self.model = self.model.partial_fit(X, y, classes=classes)

    def predict(self, X):
        return self.model.predict(X)
