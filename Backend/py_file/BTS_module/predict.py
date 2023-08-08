import os
import numpy as np
import pandas as pd
import librosa
import cv2
import matplotlib.pyplot as plt
from tensorflow import keras

def Image_data_prediction(hyperparameter, hyperparameter_name ,pred_file_path):
    model = keras.models.load_model("/BTS/training_model/training_model(" + hyperparameter_name + ").h5")
    df = pd.DataFrame(columns = ['file_name', 'prediction'])
    df.to_csv("/BTS/result/Predict_result/(result){}.csv".format(hyperparameter_name), index = False)

    for filename in os.listdir(pred_file_path):
        if hyperparameter["color"] == "rgb":
            img = cv2.imread(os.path.join(pred_file_path, filename), cv2.IMREAD_COLOR)
        else :
            img = cv2.imread(os.path.join(pred_file_path, filename), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, dsize = (hyperparameter["size_width"], hyperparameter["size_height"]))
        img = img/255
        if hyperparameter["color"] == "rgb":
            img = img.reshape(-1, hyperparameter["size_width"], hyperparameter["size_height"], 3)
        else :
            img = img.reshape(-1, hyperparameter["size_width"], hyperparameter["size_height"], 1)
        score = model.predict(img)
        print(filename, score)
        if score < 0.5:
            df = pd.DataFrame([[filename[:-4], 0]], columns = ['file_name', 'prediction'])
        else :
            df = pd.DataFrame([[filename[:-4], 1]], columns = ['file_name', 'prediction'])
        df.to_csv("/BTS/result/Predict_result/(result){}.csv".format(hyperparameter_name), mode = 'a', header = False, index = False)
    
    df = pd.DataFrame([["end", "end"]], columns = ['file_name', 'prediction'])
    df.to_csv("/BTS/result/Predict_result/(result){}.csv".format(hyperparameter_name), mode = 'a', header = False, index = False)

def export_Mel_Spectrogram(Sound_path, Image_path):

    frame_length = 0.025
    frame_stride = 0.010

    file_name = []

    for filename in os.listdir(Sound_path):
        file_name.append(filename)
        y, sr = librosa.load(os.path.join(Sound_path, filename), sr = None)

        input_nfft = int(round(sr*frame_length))
        input_stride = int(round(sr*frame_stride))

        S = librosa.feature.melspectrogram(y = y, n_mels = 40, n_fft = input_nfft, hop_length = input_stride)

        plt.figure(figsize=(10, 4))
        librosa.display.specshow(librosa.power_to_db(S, ref=np.max), sr = sr, hop_length=input_stride)
        plt.axis('off')
        plt.savefig(Image_path+filename+'.png', bbox_inches = 'tight', pad_inches = 0)
        plt.close()
    return file_name