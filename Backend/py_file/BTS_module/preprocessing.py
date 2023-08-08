import pandas as pd
import numpy as np
import json
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import splitfolders
import shutil

from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def json_path(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data

# LSTM
def split_train_test(df):
    df.drop(columns = ["Time"], axis = 1, inplace = True)
    train_data = df[:round(len(df)*0.7)]
    train_data = train_data[train_data['label'] == 0]
    test_data = df[round(len(df)*0.7):]
    return train_data, test_data

def LSTM_train_data(train_data):
    X_train = train_data.drop([train_data.columns[-1]], axis = 1)
    X_train_column = X_train.columns
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    y_train = list(train_data['label'])
    return X_train, y_train, X_train_column

def LSTM_test_data(test_data):
    X_test = test_data.drop([test_data.columns[-1]], axis = 1)
    scaler = StandardScaler()
    X_test = scaler.fit_transform(X_test)
    y_test = list(test_data['label'])
    return X_test, y_test

def reshape(X_train, X_test):
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    return X_train, X_test

# IForest
def pick_column(df):
    
    y_data = df['label']
    X_data = df.drop(['Time', 'label'], axis=1)

    return X_data, y_data

# Image
def Image_Data_Generator(train_path, hyperparameter, My_Seed):
    data_generator = ImageDataGenerator(rescale = 1/255)

    train_generator = data_generator.flow_from_directory(train_path, target_size = (hyperparameter["size_width"], hyperparameter["size_height"]), 
    color_mode = hyperparameter["color"], class_mode = 'binary', batch_size = 1)
    
    n_img = train_generator.n
    steps = n_img//1

    imgs, labels = [], []
    
    for i in range(steps):
        a, b = train_generator.next()
        imgs.extend(a)
        labels.extend(b)

    X_data = np.asarray(imgs)
    y_data = np.asarray(labels)

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.2, random_state = My_Seed, stratify = y_data)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size = 0.5, random_state = My_Seed, stratify = y_test)
   
    return X_train, X_val, X_test, y_train, y_val, y_test

# Sound
def export_Mel_Spectrogram(Sound_path, Image_path):

    frame_length = 0.025
    frame_stride = 0.010

    for filename in os.listdir(Sound_path):
        y, sr = librosa.load(os.path.join(Sound_path, filename), sr = None)

        input_nfft = int(round(sr*frame_length))
        input_stride = int(round(sr*frame_stride))

        S = librosa.feature.melspectrogram(y = y, n_mels = 40, n_fft = input_nfft, hop_length = input_stride)

        plt.figure(figsize=(10, 4))
        librosa.display.specshow(librosa.power_to_db(S, ref=np.max), sr = sr, hop_length=input_stride)
        plt.axis('off')
        plt.savefig(Image_path+filename+'.png', bbox_inches = 'tight', pad_inches = 0)
        plt.close()

def Image_Data_split(image_path, split_path):
    splitfolders.ratio(image_path,
    split_path, ratio = (0.8, 0.1, 0.1))

def Sound_Data_Generator(train_path, validation_path, test_path, hyperparameter):
    data_generator = ImageDataGenerator(rescale = 1/255)

    train_generator = data_generator.flow_from_directory(
        train_path,
        target_size = (hyperparameter["size_width"], hyperparameter["size_height"]),
        color_mode = 'rgb',
        class_mode = 'binary',
        batch_size = 1)
    
    n_img = train_generator.n
    steps = n_img//1

    imgs, labels = [], []
    
    for i in range(steps):
        a, b = train_generator.next()
        imgs.extend(a)
        labels.extend(b)

    X_train = np.asarray(imgs)
    y_train = np.asarray(labels)

    validation_generator = data_generator.flow_from_directory(
        validation_path,
        target_size = (hyperparameter["size_width"], hyperparameter["size_height"]),
        color_mode = 'rgb',
        class_mode = 'binary',
        batch_size = 1)

    n_img = validation_generator.n
    steps = n_img//1

    imgs, labels = [], []

    for i in range(steps):
        a, b = validation_generator.next()
        imgs.extend(a)
        labels.extend(b)

    X_val = np.asarray(imgs)
    y_val = np.asarray(labels)

    test_generator = data_generator.flow_from_directory(
        test_path,
        target_size = (hyperparameter["size_width"], hyperparameter["size_height"]),
        color_mode = 'rgb',
        class_mode = 'binary',
        batch_size = 1)
    
    n_img = test_generator.n
    steps = n_img//1

    imgs, labels = [], []

    for i in range(steps):
        a, b = test_generator.next()
        imgs.extend(a)
        labels.extend(b)

    X_test = np.asarray(imgs)
    y_test = np.asarray(labels)

    return X_train, X_val, X_test, y_train, y_val, y_test

#remove
def remove_files(file_path):
    for filename in os.listdir(file_path):
        os.remove(os.path.join(file_path, filename))

def remove_split_folders(file_path):
    for filename in os.listdir(file_path):
        shutil.rmtree(os.path.join(file_path, filename))