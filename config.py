import os
#Author : @LLHyun  dns05018@kakao.com
#Last Update : 2023. 01. 05
#This is config.py
#Backend : Python, FastAPI
#Frontend : Streamlit
#Python version : 3.9
#with Docker, Tensorflow, Keras
#
#
Backend_Address = "http://127.0.0.1:8000"
BASE_DIR = "/home/dais02/2022BTS_DaisLab_ADP"
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#-------------------------------------------------------------------------------
###Backend Directory Setting


PATH_BASE_DIR = BASE_DIR + "/Backend"
PATH_TRAIN_CSV = "/csv/csv_file/"
PATH_PREDICT_CSV = "/csv/csv_predict_file/"
PATH_CSV_INFO = "/csv/csv_file_info/"
PATH_TRAIN_IMAGE = "/image/image_file/"
PATH_PREDICT_IMAGE = "/image/image_predict_file/"

PATH_TRAIN_SOUND = "/sound/sound_file/"
PATH_PREDICT_SOUND = "/sound/sound_predict_file/"

PATH_PROCESS_LOG = "/process_log/"
PATH_MODEL = "/training_model/"
PATH_PREDICT_RESULT = "/result/Predict_result/"


PATH_PARAMS = "/parameter(json)/"

#폴더가 없으면 생성
if not os.path.isdir(PATH_BASE_DIR):
    os.makedirs(PATH_BASE_DIR, exist_ok=True)
if not os.path.isdir(PATH_TRAIN_CSV):
    os.makedirs(PATH_BASE_DIR + PATH_TRAIN_CSV, exist_ok=True)
if not os.path.isdir(PATH_PREDICT_CSV):
    os.makedirs(PATH_BASE_DIR + PATH_PREDICT_CSV, exist_ok=True)
if not os.path.isdir(PATH_CSV_INFO):
    os.makedirs(PATH_BASE_DIR + PATH_CSV_INFO, exist_ok=True)
if not os.path.isdir(PATH_TRAIN_IMAGE):
    os.makedirs(PATH_BASE_DIR + PATH_TRAIN_IMAGE, exist_ok=True)
if not os.path.isdir(PATH_PREDICT_IMAGE):
    os.makedirs(PATH_BASE_DIR + PATH_PREDICT_IMAGE, exist_ok=True)
if not os.path.isdir(PATH_TRAIN_SOUND):
    os.makedirs(PATH_BASE_DIR + PATH_TRAIN_SOUND, exist_ok=True)
if not os.path.isdir(PATH_PREDICT_SOUND):
    os.makedirs(PATH_BASE_DIR + PATH_PREDICT_SOUND, exist_ok=True)
if not os.path.isdir(PATH_PROCESS_LOG):
    os.makedirs(PATH_BASE_DIR + PATH_PROCESS_LOG, exist_ok=True)
if not os.path.isdir(PATH_MODEL):
    os.makedirs(PATH_BASE_DIR + PATH_MODEL, exist_ok=True)
if not os.path.isdir(PATH_PREDICT_RESULT):
    os.makedirs(PATH_BASE_DIR + PATH_PREDICT_RESULT, exist_ok=True)
if not os.path.isdir(PATH_PARAMS):
    os.makedirs(PATH_BASE_DIR + PATH_PARAMS, exist_ok=True)


#-------------------------------------------------------------------------------

SERVER_PASSWORD = ""
