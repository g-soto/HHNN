import tensorflow as tf
from tensorflow import keras
import numpy as np 
from tensorflow.keras import regularizers
import tensorflow.keras.backend as k
from HH.util import DBM

class NoTrainedError(Exception):
    pass

class OHNN:
    def __init__(self):
        self.trained = False

    def train(self, data, profits):
        # self.profits = profits
        reg_coef = 0.5
        self.model = keras.Sequential([
            keras.layers.Dense(64, activation=tf.nn.relu, input_shape=(data.shape[1],), kernel_regularizer=tf.keras.regularizers.l1_l2(reg_coef, reg_coef)),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(128, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l1_l2(reg_coef, reg_coef)),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(128, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l1_l2(reg_coef, reg_coef)),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(512, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l1_l2(reg_coef, reg_coef)),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(512, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l1_l2(reg_coef, reg_coef)),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(512, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l1_l2(reg_coef, reg_coef)),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(1024, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l1_l2(reg_coef, reg_coef)),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(1024, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l1_l2(reg_coef, reg_coef)),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(512, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l1_l2(reg_coef, reg_coef)),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(512, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l1_l2(reg_coef, reg_coef)),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(512, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l1_l2(reg_coef, reg_coef)),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(128, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l1_l2(reg_coef, reg_coef)),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(128, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l1_l2(reg_coef, reg_coef)),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(64, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l1_l2(reg_coef, reg_coef)),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(profits.shape[1])
        ])
        self.model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
            #   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            loss=lambda targets, outputs : tf.sqrt(tf.reduce_mean((targets - outputs)**2)),
            metrics=[DBM()]
              )
        self.model.fit(data, profits, epochs=2500)
        self.trained = True
        self.model.save('model.tf')

    def predict(self, data):
        if self.trained:
            return tf.argmin(self.model.predict(data), axis = 1)
            # probability_model = tf.keras.Sequential([self.model,tf.keras.layers.Softmax()])
            # predictions = probability_model.predict(data)
            # return np.argmax(predictions, axis=1)
        else:
            raise NoTrainedError('The network has not been trained yet')

    def evaluate(self, data, labels):
        if self.trained:
            return self.model.evaluate(data,  labels, verbose=1)
        else:
            raise NoTrainedError('The network has not been trained yet')

    def load_model(self):
        self.trained = True
        self.model = tf.keras.models.load_model('model.tf')
