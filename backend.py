import shutil
from typing import List
import os,sys
from fastapi import FastAPI,UploadFile,File,Request
import csv
import pandas as pd
import subprocess
import requests
import json
from fastapi.responses import FileResponse
import paramiko
import pydantic
import datetime
import config
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
app = FastAPI()

PASSWORD = config.SERVER_PASSWORD

PATH_BASE_DIR = config.PATH_BASE_DIR
PATH_TRAIN_CSV = PATH_BASE_DIR + config.PATH_TRAIN_CSV
PATH_PREDICT_CSV = PATH_BASE_DIR + config.PATH_PREDICT_CSV
PATH_CSV_INFO = PATH_BASE_DIR + config.PATH_CSV_INFO
PATH_TRAIN_IMAGE = PATH_BASE_DIR +  config.PATH_TRAIN_IMAGE
PATH_PREDICT_IMAGE = PATH_BASE_DIR + config.PATH_PREDICT_IMAGE
PATH_TRAIN_SOUND = PATH_BASE_DIR + config.PATH_TRAIN_SOUND
PATH_PREDICT_SOUND = PATH_BASE_DIR + config.PATH_PREDICT_SOUND
PATH_PROCESS_LOG = PATH_BASE_DIR + config.PATH_PROCESS_LOG
PATH_MODEL = PATH_BASE_DIR + config.PATH_MODEL
PATH_PREDICT_RESULT = PATH_BASE_DIR + config.PATH_PREDICT_RESULT
PATH_PARAMS = PATH_BASE_DIR + config.PATH_PARAMS

#-------------------------------------------------------------------------------
CSV = pd.DataFrame()
CSV_PRE = pd.DataFrame()
INDEX = 1
DETECTION = True
DOCKER_RUN = None
ML_MODEL = None
DATA_INFO = None


@app.post("/init")
async def init():
    CSV = pd.DataFrame()
    INDEX = 1
    DETECTION = False
    DOCKER_RUN = None
    return {"message": "init"}

#just print what i got from the client


@app.post("/realtime/postdata")
async def realdata1(parameter: dict):
    global INDEX, PATH_MODEL, DETECTION, CSV, DATA_INFO
    if DETECTION == True:
        CSV = pd.DataFrame(columns=parameter.keys())
        CSV.loc[INDEX] = parameter.values()



@app.get("/realtime/getdata")
async def realdata2():
    global CSV
    if CSV.empty == False:
        json = CSV.tail(1).to_json(orient='records')
        return json
    else:
        return "No data"

@app.post("/realtime/detection")
async def detection(info: str):
    global DETECTION, CSV, INDEX, DATA_INFO
    df = pd.DataFrame(json.loads(info))
    df = df.astype(float)
    
    for i in df.columns:
        if i in DATA_INFO.index:
            df[i] = (float(CSV.tail(1)[i].values[0]) - DATA_INFO["mean"][i]) / DATA_INFO["std"][i]

    df = df.values.reshape(1,1,df.shape[1])
    result = ML_MODEL.predict(df)
    result = result.reshape(-1)
    return {"result": str(result[0])}
    


    
    


@app.get("/hyun/init_realtime")
async def init_realtime(model:str):
    global INDEX, ML_MODEL, DETECTION, DATA_INFO, CSV_PRE
    INDEX = 0
    # if model = .h5
    print(model)
    if model[-3:] == ".h5":
        ML_MODEL = tf.keras.models.load_model(PATH_MODEL+"training_model({}).h5".format(model[:-3]))
        DATA_INFO = pd.read_csv(PATH_CSV_INFO+"/{}_mean_std.csv".format(model[:-3]))
        json = DATA_INFO.to_json(orient='records')
        DATA_INFO = DATA_INFO.set_index("Unnamed: 0")
        DATA_INFO = DATA_INFO.drop(DATA_INFO.index[-1])
        CSV_PRE = pd.DataFrame(columns=DATA_INFO.index)
        DETECTION = True    

    elif model[-4:] == ".pkl":
        ML_MODEL = joblib.load("/BTS/training_model/training_model(" + hyperparameter_file_name + ").pkl")
        DATA_INFO = pd.read_csv(PATH_CSV_INFO+"/{}_mean_std.csv".format(model[:-4]), index_col=0)
        json = DATA_INFO.to_json(orient='records')
        DATA_INFO = DATA_INFO.set_index("Unnamed: 0")
        DATA_INFO = DATA_INFO.drop(DATA_INFO.index[-1])
        CSV_PRE = pd.DataFrame(columns=DATA_INFO.index)
        DETECTION = True


    if DETECTION == True:
        return json

@app.post("/hyun/stop_realtime")
async def init_realtime(model:str):
    global INDEX, DETECTION
    INDEX = 0
    DETECTION = False
    return "데이터 수집이 중지되었습니다."




@app.post("/hyun/uploadfiles")
async def upload_all_files(file: UploadFile = File(...), dir: str = None, name: str = None):
    UPLOAD_DIRECTORY = dir
    contents = await file.read()
    print(file)
    with open(os.path.join(UPLOAD_DIRECTORY, name), "wb") as fp:
        fp.write(contents)
    return {"filenames": name}

@app.post("/hyun/hyperparameter")
async def hyperparameter(model: str = None, epochs: int = None, batch_size: int = None, learning_rate: float = None, color:str=None, size_width:int=None, size_height:int=None, filename: str = None):
    patch = PATH_PARAMS + filename + ".json"
    jsondata ={
    "model" : model,
    "epochs" : epochs, 
    "batch_size" : batch_size, 
    "learning_rate" : learning_rate, 
    "color" : color,
    "size_width" : size_width,
    "size_height" : size_height
    }
    
    with open(patch, 'w', encoding='utf-8') as file:
      json.dump(jsondata, file)
    
    return {"filenames": filename}

@app.get("/hyun/processlog")
async def processlog(filename: str = None):
    #파일이 없으면 계속 검사한다.
    #파일이 있으면 파일을 읽어서 return
    if os.path.isfile(PATH_PROCESS_LOG + filename + ".csv"):
        history = pd.read_csv(PATH_PROCESS_LOG + filename + ".csv").tail(1).to_json(orient='records')
        print(history)
        return history
    
@app.get("/hyun/model_list")
async def model_list():
    try:
        model_list = os.listdir(PATH_MODEL)
        for i in range(len(model_list)):
            model_list[i] = model_list[i].replace("training_model(", "")
            model_list[i] = model_list[i].replace(")", "")
        return model_list
    except:
        pass
        
@app.get("/hyun/model_info")
async def model_info(model:str = None):
    loaded_model = tf.keras.models.load_model(PATH_MODEL+'training_model({}).h5'.format(model[:-3]))
    model_info = loaded_model.to_json(indent=4)
    return model_info

@app.post("/hyun/run_model")
async def run_model(filename: str = None, model:str = None):
    command = "echo '{}' | sudo -S docker run -i --rm --gpus ''device=1'' -v {}:/BTS test:0.1 BTS/py_file/test.py 'BTS/parameter(json)/{}.json' &".format(PASSWORD,PATH_BASE_DIR, filename)
    subprocess.Popen(command, shell=True)

    return {"filenames": filename}

@app.post("/hyun/stop_model")
async def stop_model():
    global DOCKER_RUN
    subprocess.Popen.kill(DOCKER_RUN)
    return "모델이 중지되었습니다."

@app.post("/hyun/prepare_run_model")
async def prepare_run_model(filename: str = None):
    #if test_result.csv exist, delete fileㅍ파
    try:
        os.remove(PATH_PROCESS_LOG+"{}.csv".format(filename))
    except:
        pass

@app.get("/hyun/predict_count_files")
async def predict_count_files():
    count = len(os.listdir(PATH_PREDICT_IMAGE))
    count2 = len(os.listdir(PATH_PREDICT_SOUND))
    count3 = len(os.listdir(PATH_PREDICT_CSV))
    #return biggest count
    if count > count2:
        if count > count3:
            return count
        else:
            return count3
    else:
        if count2 > count3:
            return count2
        else:
            return count3

@app.get("/hyun/predict_process")
async def predict_process(filename: str = None):
    #파일이 없으면 계속 검사한다.
    #파일이 있으면 파일을 읽어서 return
    if os.path.isfile(PATH_PREDICT_RESULT + "(result){}.csv".format(filename)):
        history = pd.read_csv(PATH_PREDICT_RESULT + "(result){}.csv".format(filename)).tail(1).to_json(orient='records')
        print(history)
        return history


@app.post("/hyun/predict_with_model")
async def run_model(model:str):
    print(model)
    command = "echo '{}' | sudo -S docker run -i --rm --gpus ''device=1'' -v {}:/BTS predict:0.1 BTS/py_file/predict.py 'BTS/parameter(json)/{}.json' &".format(PASSWORD, PATH_BASE_DIR ,model)
    
    subprocess.Popen(command, shell=True)

    return {"filenames": model}
    

@app.post("/hyun/prepare_predict")
async def prepare_predict(filename: str = None):
    #if test_result.csv exist, delete file
    try:
        os.remove(PATH_PREDICT_RESULT+"(result){}.csv".format(filename))
    except:
        pass

@app.get("/hyun/predict_result")
async def predict_result(filename: str = None):
    #파일이 없으면 계속 검사한다.
    #파일이 있으면 파일을 읽어서 return
    if os.path.isfile(PATH_PREDICT_RESULT + "(result){}.csv".format(filename)):
        history = pd.read_csv(PATH_PREDICT_RESULT + "(result){}.csv".format(filename))
        return history.to_json(orient="records")

            

@app.post("/getparameter")
async def hyperparameter(model: str = None, epochs: int = None, batch_size: int = None, learning_rate: float = None,color: str = None, size_width : int = None, size_height : int = None, filename: str = None):
    patch = PATH_PARAMS+filename+".json"
    jsondata ={
    "model" : model,
    "epochs" : epochs, 
    "batch_size" : batch_size, 
    "learning_rate" : learning_rate,
    "color" : color,
    "size_width" : size_width,
    "size_height" : size_height,
    "filename" : filename
    }
    with open(patch, 'w') as make_file:
        json.dump(jsondata, make_file, indent="\t", ensure_ascii=False)
    return jsondata