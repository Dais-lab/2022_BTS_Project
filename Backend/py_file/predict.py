import sys
import os
import pandas as pd
import joblib
import warnings
warnings.filterwarnings(action='ignore')

from pathlib import Path

from tensorflow import keras
from BTS_module.predict import Image_data_prediction, export_Mel_Spectrogram
from BTS_module.preprocessing import json_path, remove_files, remove_split_folders
from sklearn.preprocessing import StandardScaler

My_Seed = 72

hyperparameter_path = sys.argv[1]
hyperparameter_file_name = Path(hyperparameter_path).stem
hyperparameter = json_path(hyperparameter_path)

if hyperparameter["model"] == "LSTM":
    model = keras.models.load_model("/BTS/training_model/training_model(" + hyperparameter_file_name + ").h5")
    train_path = "/BTS/csv/csv_predict_file/"
    for filename in os.listdir(train_path):
        data_path = train_data_path = os.path.join(train_path + filename)
        print(data_path)
    df = pd.read_csv(data_path)
    X_data = df.drop(["Time"], axis = 1)
    scaler = StandardScaler()
    X_data = scaler.fit_transform(X_data)
    X_data = X_data.reshape(X_data.shape[0], 1, X_data.shape[1])
    y_pred = model.predict(X_data)
    y_pred = y_pred.reshape(-1)
    temp = []
    for i in range(len(y_pred)):
        if y_pred[i] > 0.5:
            temp.append(1)
        else:
            temp.append(0)
    y_pred = temp
    df1 = pd.DataFrame(y_pred)
    df1.to_csv("/BTS/result/Predict_result/(result)" + hyperparameter_file_name + ".csv", index = False)
    remove_files("/BTS/csv/csv_predict_file")

elif hyperparameter["model"] == "IForest":
    model = joblib.load("/BTS/training_model/training_model(" + hyperparameter_file_name + ").pkl")
    train_path = "/BTS/csv/csv_predict_file/"
    for filename in os.listdir(train_path):
        data_path = os.path.join(train_path + filename)
    df = pd.read_csv(data_path) 
    X_data = df.drop(["Time"], axis = 1)
    y_pred = model.predict(X_data)
    row_data = {"Time" : df["Time"], "result" : y_pred}
    df = pd.DataFrame(row_data)
    df.to_csv("/BTS/result/Predict_result/(result)" + hyperparameter_file_name + ".csv", index = False)
    remove_files("/BTS/csv/csv_predict_file")

else:
    len_sound_path = "/BTS/sound/sound_file/Accept"

    if len(os.listdir(len_sound_path)) == 0:
        jpg_path = "/BTS/image/image_predict_file"
        model = keras.models.load_model("/BTS/training_model/training_model(" + hyperparameter_file_name + ").h5")
        Image_data_prediction(hyperparameter, hyperparameter_file_name ,"/BTS/image/image_predict_file")
        remove_files("/BTS/image/image_predict_file")

    else:
        Sound_predict_path = "/BTS/sound/sound_predict_file"
        Jpg_predict_path = "/BTS/sound/sound_predict_file(jpg)/"

        file_name = export_Mel_Spectrogram(Sound_predict_path, Jpg_predict_path)
        Image_data_prediction(hyperparameter, hyperparameter_file_name, Jpg_predict_path)
        remove_files("/BTS/sound/image_predict_file")
        remove_files("/BTS/sound/image_predict_file(jpg)")