import datetime
import sys
import os
import random
import csv
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.preprocessing import StandardScaler
from pandas import Series
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.layers import Activation, Dense, Dropout, LSTM


class Trader:

    def __init__(self):
        self.prices = []
        self.window_size = 20
        self.epochs = 100
        self.neurons = 100
        self.lr = 0.001
        self.model_path = '/var/keras/model.h5'

    def create_model(self, layer1, layer2, layer3, layer4, lr):
        model = Sequential()
        model.add(LSTM(
            input_shape=(layer2, layer1),
            units=layer2,
            return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(layer3, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=layer4))
        model.add(Activation("linear"))
        optimizer = Adam(lr=lr)
        model.compile(loss="mse", optimizer=optimizer)
        return model

    def make_batches(self, batch_size, data_set):
        x = [data_set[i:batch_size + i]
             for i in range(0, len(data_set) - batch_size)]
        y = [data_set[batch_size + i]
             for i in range(0, len(data_set) - batch_size)]
        return x, y

    def plot_results_multiple(self, predicted_data, true_data, prediction_len):
        fig = plt.figure(facecolor='white')
        ax = fig.add_subplot(111)
        ax.plot(true_data, label='Actual Data')
        plt.legend()
        # Pad the list of predictions to shift it in the graph to it's correct start
        print(predicted_data)
        for i, data in enumerate(predicted_data):
            print(data)
            padding = [None for p in range(i * prediction_len)]
            plt.plot(padding + data, label='Prediction')
        plt.show()
        return fig

    def predict_sequences_multiple(self, model, data, window_size):
        # Predict sequence of window_size steps before shifting prediction run forward by window_size steps
        prediction_seqs = []
        # print('length of data', len(data), len(data[0]), '\n=========data=======\n', data)
        for i in range(int(len(data) / window_size)):
            curr_frame = data[i * window_size]
            predicted = []
            for j in range(window_size):
                predicted.append(model.predict(
                    curr_frame[np.newaxis, :, :])[0, 0])
                curr_frame = curr_frame[1:]
                curr_frame = np.insert(
                    curr_frame, [window_size - 1], predicted[-1], axis=0)
            prediction_seqs.append(predicted)
        return prediction_seqs

    def normalize_data(self, data):
        series = Series(data)
        series_values = series.values
        series_values = series_values.reshape((len(series_values), 1))
        # train the normalization
        scaler = StandardScaler()
        scaler = scaler.fit(series_values)
        standardized = scaler.transform(series_values)
        return standardized

    def build_train_and_test_data(self, data, window_size, training_pct, normal_in_window):
        test_set = []
        training_set = []
        x_train = []
        y_train = []
        x_test = []
        y_test = []
        train_array = data[:int(len(data)*training_pct)]
        test_array = data[int(len(data)*training_pct):]

        if len(train_array) > 0:
            training_set = self.normalize_data(train_array)
            x_train, y_train = self.make_batches(window_size, training_set)
        if len(test_array) > 0:
            test_set = self.normalize_data(test_array)
            x_test, y_test = self.make_batches(window_size, test_set)

        x_train = np.array(x_train, dtype=float)
        y_train = np.array(y_train, dtype=float)
        x_test = np.array(x_test, dtype=float)
        y_test = np.array(y_test, dtype=float)

        # print('x_train =====', x_train)
        # print('y_train =====', y_train)
        # print('diff: ', diff)
        return x_train, y_train, x_test, y_test

    def train(self):
        training_pct = 0.8
        val_pct = 0.2

        x_train, y_train, x_test, y_test = self.build_train_and_test_data(
            self.prices, self.window_size, training_pct, True)

        model = self.create_model(1, self.window_size, self.neurons, 1, self.lr)

        model.fit(x_train, y_train, batch_size=32, epochs=self.epochs,
                  validation_split=val_pct, shuffle=False)
        mse = model.evaluate(x_test, y_test)
        print('Accuracy/Mean Squared Error: ', mse)
        # predictions = self.predict_sequences_multiple(
        #         model, x_test, self.window_size)
        # print(len(x_test), len(y_test), len(predictions))
        # print(predictions)
        # fig = plot_results_multiple(
        #     predictions, y_test, self.window_size, name, mse, nrmsd, res)
        # fig.savefig('id%s_%s_epochs%s_ws%s_nn%s_%s' % (
        #     266, res, epochs, self.window_size, neurons))
        model.save(self.model_path)
        return model

    def predict(self, plot = False):
        if os.path.isfile(self.model_path):
            print("Loading", self.model_path)
            model = load_model(self.model_path)
        else:
            print("Training new model")
            model = self.train()
        print(model.summary())
        x_train, y_train, x_test, y_test = self.build_train_and_test_data(
            self.prices, self.window_size, 0, True)
        predictions = self.predict_sequences_multiple(model, x_test, self.window_size)
        # print('pred', len(predictions))
        # print('x_test', len(x_test))
        # print('y_test', len(y_test))
        if plot:
            self.plot_results_multiple(predictions, y_test, self.window_size)
        return predictions
