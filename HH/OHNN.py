import tensorflow as tf
from tensorflow import keras
import numpy as np 

class NoTrainedError(Exception):
    pass

class OHNN:
    def __init__(self):
        self.trained = False

    def train(self, data, labels):
        self.model = keras.Sequential([
            keras.layers.Dense(64, activation=tf.nn.relu, input_shape=(data.shape[1],)),
            keras.layers.Dense(128, activation=tf.nn.relu),
            keras.layers.Dense(128, activation=tf.nn.relu),
            keras.layers.Dense(512, activation=tf.nn.relu),
            keras.layers.Dense(128, activation=tf.nn.relu),
            keras.layers.Dense(128, activation=tf.nn.relu),
            keras.layers.Dense(64, activation=tf.nn.relu),
            keras.layers.Dense(10)
        ])
        self.model.compile(optimizer=tf.keras.optimizers.Adam(0.00001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
        self.model.fit(data, labels, epochs=150000)
        self.trained = True
        self.model.save('model.tf')

    def predict(self, data):
        if self.trained:
            probability_model = tf.keras.Sequential([self.model,tf.keras.layers.Softmax()])
            predictions = probability_model.predict(data)
            return np.argmax(predictions, axis=1)
        else:
            raise NoTrainedError('The network has not been trained yet')

    def evaluate(self, data, labels):
        return self.model.evaluate(data,  labels, verbose=2)

    def load_model(self):
        self.model = tf.keras.models.load_model('model.tf')
