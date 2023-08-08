import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

from tensorflow import keras
from keras.applications import ResNet50, VGG16, EfficientNetB0
from keras.layers import LSTM, RepeatVector, TimeDistributed, Dense, Input
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from datetime import datetime
from pyod.models.iforest import IForest

def evaluate_data(Y_test, Y_pred, hyperparameter_file_name):

    TP = 0
    FP = 0
    FN = 0
    TN = 0

    for i in range(len(Y_pred)):
        if Y_test[i] == 0 and Y_pred[i] == 0:
            TN = TN + 1
        elif Y_test[i] == 0 and Y_pred[i] == 1:
            FP = FP + 1
        elif Y_test[i] == 1 and Y_pred[i] == 0:
            FN = FN + 1
        elif Y_test[i] == 1 and Y_pred[i] == 1:
            TP = TP + 1

    row_data = {"Accuracy" : [accuracy_score(Y_test, Y_pred)],
                "F1_Score" : [f1_score(Y_test, Y_pred)],
                "Recall" : [recall_score(Y_test, Y_pred)],
                "Precision" : [precision_score(Y_test, Y_pred)],
                "TN" : [TN],
                "FP" : [FP],
                "FN" : [FN],
                "TP" : [TP]}
    df = pd.DataFrame(row_data)
    df.to_csv("/BTS/result/Test_result/" + hyperparameter_file_name + ".csv")

# LSTM
def define_LSTM_model(X_train):
    training_model = keras.models.Sequential([
        LSTM(140, activation = 'tanh', input_shape = (X_train.shape[1], X_train.shape[2]), return_sequences = True),
        LSTM(70, activation = 'tanh', return_sequences = False),
        RepeatVector(X_train.shape[1]),
        LSTM(70, activation = 'tanh' , return_sequences = True),
        LSTM(140, activation = 'tanh', return_sequences = True),
        TimeDistributed(Dense(X_train.shape[2]))
    ])
    return training_model

def LSTM_training(training_model, hyperparameter, hyperparameter_file_name, X_train, y_train):
    class CustomCallback(keras.callbacks.Callback):
        def on_train_begin(self, logs = None):
            raw_data = {'epoch' : [],
                        'train_loss' : [],
                        "train_accuracy" : [],
                        'validation_loss' : [],
                        'validation_accuracy' : [],
                        'timestamp' : []}
            df = pd.DataFrame(raw_data)
            df.to_csv("/BTS/process_log/" + hyperparameter_file_name + ".csv", index = False)
        def on_epoch_end(self, epoch, logs=None):
            now = datetime.now()
            df = pd.read_csv("/BTS/process_log/" + hyperparameter_file_name + ".csv")
            df.loc[-1]=[epoch, logs["loss"], logs["binary_accuracy"], logs["val_loss"], logs["val_binary_accuracy"], now.timestamp()]
            df.to_csv("/BTS/process_log/" + hyperparameter_file_name + ".csv", index = False)
        def on_train_end(self, epoch, logs=None):
            df = pd.read_csv("/BTS/process_log/" + hyperparameter_file_name + ".csv")
            df.loc[-1]=[hyperparameter["epochs"], 0, 0, 0, 0, 0]
            df.to_csv("/BTS/process_log/" + hyperparameter_file_name + ".csv", index = False)
    
    filename = ('/BTS/training_model/training_model(' + hyperparameter_file_name + ").h5")
    checkpoint = ModelCheckpoint(filename, monitor = 'val_loss', verbose = 1, save_best_only = True, mode = 'auto')
    earlystopping = EarlyStopping(monitor = 'val_loss', patience = 30)

    training_model.compile(optimizer = keras.optimizers.Adam(learning_rate = hyperparameter["learning_rate"]), loss = 'mse', metrics = tf.keras.metrics.BinaryAccuracy())
    training_model.fit(X_train, X_train, epochs = hyperparameter["epochs"], batch_size = hyperparameter["batch_size"], validation_split = 0.1, shuffle = False, callbacks = [checkpoint, earlystopping, CustomCallback()])

def threshold(X_train, hyperparameter_file_name, X_train_column):
    filename = "/BTS/training_model/training_model(" + hyperparameter_file_name + ").h5"
    model = keras.models.load_model(filename)
    X_pred = model.predict(X_train)
    X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
    X_pred = pd.DataFrame(X_pred, columns = X_train_column)

    scored = pd.DataFrame()
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[2])
    scored["Loss_mae"] = np.mean(np.abs(X_pred - X_train), axis = 1)
    threshold = np.percentile(scored, 100)
    return threshold

def LSTM_test(hyperparameter_file_name, X_test, y_test, X_train_column, threshold):
    model = keras.models.load_model("/BTS/training_model/training_model(" + hyperparameter_file_name + ").h5")
    X_pred = model.predict(X_test)
    X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
    X_pred = pd.DataFrame(X_pred, columns = X_train_column)

    score = pd.DataFrame(index = X_pred.index)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[2])
    score["Loss_mae"] = np.mean(np.abs(X_pred - X_test), axis = 1)
    score['Threshold'] = threshold
    score['y_pred'] = score["Loss_mae"] > score["Threshold"]
    
    temp_list = []
    for i in range(len(score)):
        if score["y_pred"][i] == False:
            temp_list.append(0)
        else:
            temp_list.append(1)

    score["y_pred"] = temp_list
    score["y_test"] = y_test
    return score

# IForest
def IForest_training(df, X_train, y_train, hyperparameter_file_name):

    model_path = "/BTS/training_model/training_model(" + hyperparameter_file_name + ").pkl"
    contam = round(len(df.loc[df['label']==1]) / (len(df.loc[df['label']==0]) + len(df.loc[df['label']==1])), 2)
    model = IForest(contamination = contam, random_state=777)
    model = model.fit(X_train, y_train)
    joblib.dump(model, model_path)

    return model

def IForest_test(X_test, y_test, hyperparameter_file_name):

    model = joblib.load("/BTS/training_model/training_model(" + hyperparameter_file_name + ").pkl")
    y_pred = model.predict(X_test)
    row_data = {"y_test" : y_test, "y_pred" : y_pred}
    score = pd.DataFrame(row_data)
    return score

# Image
def define_image_model(hyperparameter):
    if hyperparameter["color"] == "rgb":
        input_tensor = Input(shape = (hyperparameter["size_width"], hyperparameter["size_height"], 3))
    else:
        input_tensor = Input(shape = (hyperparameter["size_width"], hyperparameter["size_height"], 1))

    if hyperparameter["model"] == "CNN" and hyperparameter["color"] == "rgb":
        input_CNN_tensor = (hyperparameter["size_width"], hyperparameter["size_height"], 3)
    else :
        input_CNN_tensor = (hyperparameter["size_width"], hyperparameter["size_height"], 1)

    if hyperparameter["model"] == "CNN":
        training_model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_CNN_tensor),
            tf.keras.layers.MaxPool2D(2, 2),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPool2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPool2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPool2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPool2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')])

    elif hyperparameter["model"] == "VGG16":
        training_model = VGG16(weights = None, include_top = False, input_tensor = input_tensor)

        layer_dict = dict([(layer.name, layer) for layer in training_model.layers])

        x = layer_dict['block5_pool'].output
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(512, activation = "relu")(x)
        x = tf.keras.layers.Dense(1, activation = 'sigmoid')(x)

        training_model = Model(inputs = training_model.input, outputs = x)
    
    elif hyperparameter["model"] == "ResNet50":
        training_model = ResNet50(weights = None, include_top = False, input_tensor = input_tensor)

        layer_dict = dict([(layer.name, layer) for layer in training_model.layers])
        x = layer_dict['conv5_block3_out'].output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(512, activation = "relu")(x)
        x = tf.keras.layers.Dense(1, activation = 'sigmoid')(x)

        training_model = Model(inputs = training_model.input, outputs = x)
    
    elif hyperparameter["model"] == "EfficientNetB0":
        training_model = EfficientNetB0(weights = None, include_top = False, input_tensor = input_tensor)
        
        layer_dict = dict([(layer.name, layer) for layer in training_model.layers])
        
        x = layer_dict['top_activation'].output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(512, activation = "relu")(x)
        x = tf.keras.layers.Dense(1, activation = 'sigmoid')(x)

        training_model = Model(inputs = training_model.input, outputs = x)

    return training_model

def image_training(training_model, hyperparameter, hyperparameter_file_name, X_train, X_val, y_train, y_val):

    class CustomCallback(keras.callbacks.Callback):
            def on_train_begin(self, logs = None):
                raw_data = {'epoch' : [],
                            'train_loss' : [],
                            'train_accuracy' : [],
                            'validation_loss' : [],
                            'validation_accuracy': [],
                            'timestamp' : []}
                df = pd.DataFrame(raw_data)
                df.to_csv("/BTS/process_log/" + hyperparameter_file_name + ".csv", index = False)
            def on_epoch_end(self, epoch, logs=None):
                now = datetime.now()
                df = pd.read_csv("/BTS/process_log/" + hyperparameter_file_name + ".csv")
                df.loc[-1]=[epoch, logs["loss"], logs["binary_accuracy"], logs["val_loss"], logs["val_binary_accuracy"], now.timestamp()]
                df.to_csv("/BTS/process_log/" + hyperparameter_file_name + ".csv", index = False)
            def on_train_end(self, epoch, logs=None):
                df = pd.read_csv("/BTS/process_log/" + hyperparameter_file_name + ".csv")
                df.loc[-1]=[hyperparameter["epochs"], 0, 0, 0, 0, 0]
                df.to_csv("/BTS/process_log/" + hyperparameter_file_name + ".csv", index = False)
    
    filename = ('/BTS/training_model/' + "training_model(" + hyperparameter_file_name + ").h5")
    checkpoint = ModelCheckpoint(filename, monitor = 'val_loss', verbose = 1, save_best_only = True, mode = 'auto')
    earlystopping = EarlyStopping(monitor = 'val_loss', patience = 15)

    training_model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = hyperparameter["learning_rate"]),
    loss = tf.keras.losses.BinaryCrossentropy(),
    metrics = tf.keras.metrics.BinaryAccuracy())

    training_model.fit(X_train, y_train, validation_data = (X_val, y_val), batch_size = hyperparameter["batch_size"], epochs = hyperparameter["epochs"], callbacks = [checkpoint, earlystopping, CustomCallback()])

def image_test(hyperparameter_file_name, X_test, y_test):
    model_path = "/BTS/training_model/training_model(" + hyperparameter_file_name + ").h5"
    model = keras.models.load_model(model_path)
    y_pred = model.predict(X_test)
    temp = []
    for i in range(len(y_pred)):
        if y_pred[i] > 0.5:
            temp.append(1)
        else:
            temp.append(0)
    y_pred = temp
    row_data = {"y_test" : y_test, "y_pred" : y_pred}
    score = pd.DataFrame(row_data)
    return score