# from tensorflow.keras.layers import Dense, InputLayer, Dropout
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.regularizers import L1



# class UnfairNN:
#     def __init__(self, epochs, lr, batch_size, xdim, ydim):
#         self.batch_size = batch_size
#         self.model = Sequential(
#             [   
#                 # InputLayer(input_shape=(None,)),
#                 Dense(xdim, activation='leaky_relu', kernel_initializer='ones', bias_initializer='ones', kernel_regularizer=L1(0.01)),
#                 # Dropout(0.75),
#                 # Dense(int(0.75*xdim), activation='leaky_relu', kernel_initializer='ones', bias_initializer='ones', kernel_regularizer=L1(0.01)),
#                 # Dropout(0.5),
#                 # Dense(int(0.5*xdim), activation='leaky_relu', kernel_initializer='ones', bias_initializer='ones', kernel_regularizer=L1(0.01)),
#                 # Dropout(0.25),
#                 Dense(int(0.25*xdim), activation='leaky_relu', kernel_initializer='ones', bias_initializer='ones', kernel_regularizer=L1(0.01)),
#                 Dropout(0.5),
#                 Dense(int(ydim*1.5), activation='leaky_relu', kernel_initializer='ones', bias_initializer='ones', kernel_regularizer=L1(0.01)),
#                 Dropout(0.5),
#                 Dense(ydim, activation='sigmoid')    
#             ]
#         )
#         self.epochs = epochs
#         self.lr = lr
#         self.model.compile(
#             optimizer='adam',
#             loss='binary_crossentropy',
#             metrics=['accuracy']
#         )
    
#     def fit(self, X, y):
#         return self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size)

#     def predict(self, X):
#         return self.model.predict(X)

from sklearn.neural_network import MLPClassifier

class UnfairNN:
    def __init__(self, epochs, lr, batch_size, xdim, ydim):
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.model = MLPClassifier(
            hidden_layer_sizes=(xdim, int(0.75*xdim), int(0.5*xdim), int(0.25*xdim), int(2*ydim), ydim),
            activation='relu',
            solver='adam',
            batch_size=self.batch_size, 
            learning_rate='constant', 
            learning_rate_init=self.lr,
            max_iter=self.epochs,
            random_state=42
        )
    
    def fit(self, X, y):
        self.model = self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
