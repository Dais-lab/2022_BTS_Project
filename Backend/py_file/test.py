import sys
import os
import pandas as pd
import warnings
warnings.filterwarnings(action='ignore')

from pathlib import Path

from sklearn.model_selection import train_test_split
from BTS_module.preprocessing import Image_Data_split, LSTM_test_data, LSTM_train_data, Sound_Data_Generator, export_Mel_Spectrogram, json_path, pick_column, reshape, split_train_test, Image_Data_Generator, remove_files, remove_split_folders
from BTS_module.test import IForest_test, IForest_training, LSTM_training, define_LSTM_model, define_image_model, evaluate_data, image_training, threshold, LSTM_test, image_test

My_Seed = 72

hyperparameter_path = sys.argv[1]
hyperparameter_file_name = Path(hyperparameter_path).stem
hyperparameter = json_path(hyperparameter_path)

if hyperparameter["model"] == "LSTM":
    
    train_path = "/BTS/csv/csv_file/"
    for filename in os.listdir(train_path):
        data_path = os.path.join(train_path + filename)

    df = pd.read_csv(data_path)
    #calc df mean, std 
    df_mean_std = pd.concat([df.mean(), df.std()], axis=1)
    df_mean_std.columns = ['mean', 'std']
    df_mean_std = df_mean_std.iloc[:-1]
    df_mean_std.to_csv("/BTS/csv/csv_file_info/{}_mean_std.csv".format(hyperparameter_file_name))

    train_data, test_data = split_train_test(df)
    X_train, y_train, X_train_column = LSTM_train_data(train_data)
    X_test, y_test = LSTM_test_data(test_data)
    X_train, X_test = reshape(X_train, X_test)
    model = define_LSTM_model(X_train)
    LSTM_training(model, hyperparameter, hyperparameter_file_name, X_train, y_train)
    thresholds = threshold(X_train, hyperparameter_file_name, X_train_column)
    score = LSTM_test(hyperparameter_file_name, X_test, y_test, X_train_column, thresholds)
    evaluate_data(list(score["y_test"]), list(score["y_pred"]), hyperparameter_file_name)
    remove_files("/BTS/csv/csv_file")

elif hyperparameter["model"] == "IForest":

    train_path = "/BTS/csv/csv_file/"
    for filename in os.listdir(train_path):
        data_path = os.path.join(train_path + filename)
    
    df = pd.read_csv(data_path)
    X_data, y_data = pick_column(df)
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.2, random_state = My_Seed)
    IForest_training(df, X_train, y_train, hyperparameter_file_name)
    raw_data = {'epoch' : [],
                'train_loss' : [],
                "train_accuracy" : [],
                'validation_loss' : [],
                'validation_accuracy' : [],
                'timestamp' : []}
    df = pd.DataFrame(raw_data)
    df.loc[-1]=[0, 0, 0, 0, 0, 0]
    df.to_csv("/BTS/process_log/" + hyperparameter_file_name + ".csv", index = False)
    score = IForest_test(X_test, y_test, hyperparameter_file_name)
    evaluate_data(list(score["y_test"]), list(score["y_pred"]), hyperparameter_file_name)
    remove_files("/BTS/csv/csv_file")

else:
    len_sound_path = "/BTS/sound/sound_file/Accept"

    if len(os.listdir(len_sound_path)) == 0:
        jpg_path = "/BTS/image/image_file"
        X_train, X_val, X_test, y_train, y_val, y_test = Image_Data_Generator(jpg_path, hyperparameter, 72)
        model = define_image_model(hyperparameter)
        image_training(model, hyperparameter, hyperparameter_file_name, X_train, X_val, y_train, y_val)
        score = image_test(hyperparameter_file_name, X_test, y_test)
        evaluate_data(list(score["y_test"]), list(score["y_pred"]), hyperparameter_file_name)
        remove_files("/BTS/image_file/Accept")
        remove_files("/BTS/image_file/Reject")

    else:
        Wav_accept_path = '/BTS/sound/sound_file/Accept'
        Wav_reject_path = '/BTS/sound/sound_file/Reject'

        jpg_path = "/BTS/sound/sound_file(jpg)"
        jpg_accept_path = '/BTS/sound/sound_file(jpg)/Accept/'
        jpg_reject_path = '/BTS/sound/sound_file(jpg)/Reject/'

        data_path = "/BTS/sound/sound_file(split)"
        train_path = '/BTS/sound/sound_file(split)/train'
        validation_path = '/BTS/sound/sound_file(split)/val'
        test_path = '/BTS/sound/sound_file(split)/test'
        export_Mel_Spectrogram(Wav_accept_path, jpg_accept_path)
        export_Mel_Spectrogram(Wav_reject_path, jpg_reject_path)
        Image_Data_split(jpg_path, data_path)
        X_train, X_val, X_test, y_train, y_val, y_test = Sound_Data_Generator(train_path, validation_path , test_path, hyperparameter)
        model = define_image_model(hyperparameter)
        image_training(model, hyperparameter, hyperparameter_file_name, X_train, X_val, y_train, y_val)
        score = image_test(hyperparameter_file_name, X_test, y_test)
        evaluate_data(list(score["y_test"]), list(score["y_pred"]), hyperparameter_file_name)

        remove_files("/BTS/sound_file/Accept")
        remove_files("/BTS/sound_file/Reject")
        remove_files("/BTS/sound_file(jpg)/Accept") 
        remove_files("/BTS/sound_file(jpg)/Reject") 
        remove_split_folders("/BTS/sound_file(split)")