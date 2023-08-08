import pandas as pd
import streamlit as st
import extra_streamlit_components as stx
import time
from dtale.views import startup
import streamlit.components.v1 as components
import copy
import requests
import json
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import config

#-------------------------------------------------------------------------------
###Directory Setting
BASE_DIR = config.BASE_DIR
TRAIN_CSV = config.PATH_TRAIN_CSV
PREDICT_CSV = config.PATH_PREDICT_CSV

TRAIN_IMAGE = config.PATH_TRAIN_IMAGE
PREDICT_IMAGE = config.PATH_PREDICT_IMAGE

TRAIN_SOUND = config.PATH_TRAIN_SOUND
PREDICT_SOUND = config.PATH_PREDICT_SOUND

PROCESS_LOG = config.PATH_PROCESS_LOG
MODEL = config.PATH_MODEL
PREDICT_RESULT = config.PATH_PREDICT_RESULT
PAGES_DIR = BASE_DIR + "/pages/"
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
###Backend Setting
SERVER = config.Backend_Address
#-------------------------------------------------------------------------------


def page_guide_train():
    val = stx.tab_bar(data=[
    stx.TabBarItemData(id=1, title="", description="시계열 모델"),
    stx.TabBarItemData(id=2, title="", description="이미지 모델"),
    stx.TabBarItemData(id=3, title="", description="하이퍼 파라미터"),
    stx.TabBarItemData(id=4, title="", description="모델 학습")
    ], default=1, key = "train_guide")
    if val == '1':
        html_view("pages/Time Series Model.html", 1200)
    elif val == '2':
        html_view("pages/Image Model.html", 1200)
    elif val == '3':
        html_view("pages/Hyperparameter.html", 2000)
    elif val == "4":
        html_view("pages/model train.html", 1500)
        
def page_predict():
    response = requests.get(SERVER + '/hyun/model_list')
    model_list = copy.deepcopy(response.json())
    try:
        model_select = st.session_state['params'] + ".h5"
        if model_select in model_list:
            index = model_list.index(model_select)
            temp = model_list[0]
            model_list[0] = model_list[index]
            model_list[index] = temp
        else:
            model_select = model_select[:-3] + ".pkl"
            index = model_list.index(model_select)
            temp = model_list[0]
            model_list[0] = model_list[index]
            model_list[index] = temp
    except:
        pass
    with st.sidebar:
        menu = stx.stepper_bar(steps=["길라잡이", "모델 선택", "데이터 업로드",  "데이터 예측"], is_vertical=True, lock_sequence=False)
    if menu == 0:
        html_view(PAGES_DIR+"data predict.html", 1200)

        

    elif menu == 1:
        st.subheader("모델 불러오기")
        model_select = st.selectbox("", model_list)
        st.write("선택된 모델 : ", model_select)
        
        response = requests.get(SERVER + '/hyun/model_info', {'model':model_select})
        try:
            formatted_str = json.loads(response.json())
            st.write("모델 정보")
            st.write(formatted_str["config"]["layers"])
        except Exception as e:
            st.warning("pkl 파일은 모델 정보를 불러올 수 없습니다.")
        
        if model_select[-3:] == ".h5":
            st.session_state['params'] = model_select[:-3]
        else:
            st.session_state['params'] = model_select[:-4]
        

        
        

    elif menu == 2:
        st.subheader("""테스트 데이터 업로드.""")
        upload_predict = st.file_uploader(" ", key="predict", accept_multiple_files=True)
        if len(upload_predict) > 0:
            data_preview(upload_predict[0], st)
            for i in upload_predict:
                data_upload(i, i.name, st, "predict")
                

    elif menu == 3:
        response = requests.get(SERVER + '/hyun/predict_count_files', params = {'filename':st.session_state['params']})
        count = response.json()

        st.write("예측할 데이터 개수 : ", count)

        predict = st.button("예측하기")
        message = st.empty()
        if predict:
            start = 1
            try:
                requests.post(SERVER + '/hyun/prepare_predict', params = {'filename': st.session_state['params']})
                requests.post(SERVER + '/hyun/predict_with_model', params = {'model':st.session_state.params}, timeout=0.00001)
            except requests.exceptions.ReadTimeout:
                pass
            history = pd.DataFrame(columns=['file_name', 'prediction'])
            while start == 1:
                time.sleep(0.2)
                try:
                    response = requests.get(SERVER + '/hyun/predict_process', params = {'filename':st.session_state['params']})
                    df_temp = pd.DataFrame(json.loads(response.json()))
                    history = pd.concat([history, df_temp])
                    if len(history) == 0:
                        message.warning("모델 준비 중...")
                    elif len(history) >= 1:
                        message.warning("데이터 예측 중...")
                        if history.iloc[-1, 0] == "end":
                            start = 0
                            message.success("예측 완료")
                            break

                except TypeError:
                    message.error("전처리 중...")
                    pass
            response = requests.get(SERVER + '/hyun/predict_result', params = {'filename':st.session_state['params']})
            df = pd.DataFrame(json.loads(response.json()))[:-1]
            st.table(df)
                
                    
                    

                    
            
                
                
            
    
            

            

    
    

    


        
def page_guide_preprocess():
    pass
    

def page_guide_realtime():
    pass
    




def train_data_menu(step:int):
    """
    데이터 학습 메뉴.
    """
    if step == 0:
        page_guide_train()


    elif step == 1:
        st.header("데이터 업로드")
        col1, col2 = st.columns(2)
        upload_train_True = col1.file_uploader(" ", key="train_true_file", accept_multiple_files=True)
        upload_train_False = col2.file_uploader("", key="train_false_file", accept_multiple_files=True)
        
        if len(upload_train_True) > 0:
            data = copy.deepcopy(upload_train_True[0])
            data_preview(data, col1)
            for i in upload_train_True:
                data_upload(i, i.name, col1, "train", "True")


        if len(upload_train_False) > 0:
            data = copy.deepcopy(upload_train_False[0])
            data_preview(data, col2)
            for i in upload_train_False:
                data_upload(i, i.name, col2, "train", "False")
            
            
    elif step == 2:
        try:
            hyperparameter_setting(st.session_state.DataType)
        except:
            st.warning("데이터를 업로드해주세요.")
            

    elif step == 3:
        
        model_training(st.session_state.DataType, st.session_state.params)
        
        

    


def html_view(html_path:str, height:int):
    """
    웹 페이지를 스트림릿에 띄우는 함수.
    """
    with open(html_path) as f:
        html = f.read()
    components.html(html, height=height, width=900)

def set_config():
    """
    Streamlit 기본 설정.
    """
    st.set_page_config(
        page_title="2022 BTS 이상 탐지 플랫폼",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

def set_sidebar():
    """
    Streamlit 사이드바 설정.
    """
    st.sidebar.title("2022 BTS 이상 탐지 플랫폼 🔨")

    menu = st.sidebar.selectbox("",("데이터 학습", "데이터 예측","데이터 전처리", "실시간 이상탐지", "초기화"))

    return menu

def data_preview(data:object, comp:str):
    """
    데이터 미리보기
    """
    if data is not None:
        if data.name.endswith(".csv"):
            comp.subheader("Data preview - {}".format(data.name))
            comp.write(pd.read_csv(data))
        elif data.name.endswith(".xlsx"):
            comp.subheader("Data preview - {}".format(data.name))
            comp.write(pd.read_excel(data))
        elif data.name.lower().endswith(".wav") or data.name.lower().endswith(".mp3"):
            comp.subheader("Data preview - {}".format(data.name))
            comp.audio(data, format='audio/wav')
        elif data.name.lower().endswith(".jpg") or data.name.lower().endswith(".png") or data.name.lower().endswith(".jpeg"):
            comp.subheader("Data preview - {}".format(data.name))
            comp.image(data)

def request_upload(dir:str, file:object, file_name:str, type:str, comp:str):
        if 'DataType' not in st.session_state:
            st.session_state.DataType = type
        else:
            st.session_state.DataType = type
        URL = SERVER + '/hyun/uploadfiles'
        bytes_data = file.read()
        response = requests.post(URL, files={'file':bytes_data}, params={"dir":dir, "name":file_name})
        if response.status_code == 200:
            comp.success('Upload Success : {}'.format(response.json()['filenames']))
        else:
            comp.error('Upload Fail with code : {}'.format(response.status_code))

def data_upload(file:object, file_name:str, comp:str, upload_type:str, label:str = None):
    """
    데이터 업로드.
    """
    if upload_type == 'train':
        if file_name.lower().endswith(".csv") or file_name.lower().endswith(".xlsx"):
            if label == 'True':
                request_upload(TRAIN_CSV, file, file_name, "DataFrame", comp)
            elif label == 'False':
                request_upload(TRAIN_CSV, file, file_name, "DataFrame", comp)

        elif file_name.lower().endswith(".wav") or file_name.lower().endswith(".mp3"):
            if label == 'True':
                request_upload(TRAIN_SOUND + "Accept", file, file_name, "Sound", comp)
            elif label == 'False':
                request_upload(TRAIN_SOUND + "Reject", file, file_name, "Sound", comp)

        elif file_name.lower().endswith(".jpg") or file_name.lower().endswith(".png") or file_name.lower().endswith(".jpeg"):
            if label == 'True':
                request_upload(TRAIN_IMAGE + "Accept", file, file_name, "Image", comp)
            elif label == 'False':
                request_upload(TRAIN_IMAGE + "Reject", file, file_name, "Image", comp)
            

    elif upload_type == 'predict':
        if file_name.lower().endswith(".csv") or file_name.lower().endswith(".xlsx"):
            
            request_upload(PREDICT_CSV, file, file_name, "DataFrame", comp)
            

        elif file_name.lower().endswith(".wav") or file_name.lower().endswith(".mp3"):

            request_upload(PREDICT_SOUND, file, file_name, "Sound", comp)
            

        elif file_name.lower().endswith(".jpg") or file_name.lower().endswith(".png") or file_name.lower().endswith(".jpeg"):
            
            request_upload(PREDICT_IMAGE, file, file_name, "Image", comp)
            

    

def params_upload(args:dict, file_name:str):
    """
    하이퍼파라미터 업로드.
    """
    URL = SERVER + '/getparameter'
    info = {
        "model":args['model'],
        "epochs":args['epochs'],
        "batch_size":args['batch_size'],
        "learning_rate":args['learning_rate'],
        "color":args['color'],
        "size_width":args['size_width'],
        "size_height":args['size_height'],
        "filename":file_name
    }
    response = requests.post(URL, params=info)
    if response.status_code == 200:
        st.success('Upload Success : {}.json'.format(response.json()["filename"]))
    else:
        st.error('Upload Fail with code : {}'.format(response.status_code))
    
    st.session_state['params'] = file_name
    st.session_state['epochs'] = args['epochs']

def model_summary(args:dict, comp:str):
    """
    모델 요약.
    """
    for i in args:
        if args[i] is not None:
            comp.text("{} : {}".format(i.upper(), args[i]))
        
def hyperparameter_setting(state:str = None):
    """
    업로드한 데이터 타입이 st.session_state에 저장되어 있음.
    """
    def dataframe_set():
        col1, col2 = st.columns(2, gap = "large")
        model = col1.selectbox("", ("LSTM", "IForest"))
        if model == "LSTM":
            epochs = col1.slider("Epochs", min_value=1, max_value=1000, value=10)
            batch_size = col1.slider("Batch size", min_value=4, max_value=100, value=32)
            learning_rate = col1.slider("Learning rate", min_value=0.00001, max_value=0.01, step=0.00001, format="%0.5f", value=0.0001)
            info_dict = {
                "model": model,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "color": None,
                "size_width": None,
                "size_height": None
            }
            model_summary(info_dict, col2)
            info_name = col2.text_input("하이퍼파라미터 셋 저장", "저장할 이름")
            st.session_state['epochs'] = epochs
            if col2.button("Save info"):
                params_upload(info_dict, info_name)
        elif model == "IForest":
            info_dict = {
                "model": model,
                "epochs": None,
                "batch_size": None,
                "learning_rate": None,
                "color": None,
                "size_width": None,
                "size_height": None
            }
            st.success("IForest 모델은 하이퍼파라미터를 설정할 필요가 없습니다.")
            info_name = col2.text_input("하이퍼파라미터 셋 저장", "저장할 이름")
            st.session_state['epochs'] = 0
            if col2.button("Save info"):
                params_upload(info_dict, info_name)
        st.session_state['model'] = model
            


    def sound_set():
        col1, col2 = st.columns(2, gap = "large")
        model = col1.selectbox("", ("CNN", "VGG16", "ResNet50", "EfficientNetB0"))
        epochs = col1.slider("Epochs", min_value=1, max_value=1000, value=10)
        batch_size = col1.slider("Batch size", min_value=4, max_value=100, value=32)
        learning_rate = col1.slider("Learning rate", min_value=0.00001, max_value=0.01, step=0.00001, format="%0.5f", value=0.0001)
        info_dict = {
            "model": model,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "color": "rgb",
            "size_width": 256,
            "size_height": 128
        }
        model_summary(info_dict, col2)
        info_name = col2.text_input("하이퍼파라미터 셋 저장", "저장할 이름")
        st.session_state['epochs'] = epochs
        if col2.button("Save info"):
           params_upload(info_dict, info_name)
        pass
        st.session_state['model'] = model
    def image_set():
        col1, col2 = st.columns(2, gap = "large")
        model = col1.selectbox("", ("CNN", "VGG16", "ResNet50", "EfficientNetB0"))
        color = col1.selectbox("Color", ("RGB", "Grayscale"))
        preprocess_menu = col1.multiselect("Preprocess", ("일반화", "표준화", "평탄화", "가우시안 블러", "메디안 블러", "노이즈 제거", "히스토그램 균일화"))
        epochs = col1.slider("Epochs", min_value=1, max_value=1000, value=10)
        batch_size = col1.slider("Batch size", min_value=4, max_value=100, value=32)
        learning_rate = col1.slider("Learning rate", min_value=0.00001, max_value=0.01, step=0.00001, format="%0.5f", value=0.0001)
    
        size_width = col1.select_slider("Width", options=[128, 256, 512])
        size_height = col1.select_slider("Height", options=[128, 256, 512])
        
        info_dict = {
            "model": model,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "color": color.lower(),
            "size_width": size_width,
            "size_height": size_height,
            "preprocess": preprocess_menu
        }
        model_summary(info_dict, col2)
        info_name = col2.text_input("하이퍼파라미터 셋 저장", "저장할 이름")
        st.session_state['epochs'] = epochs
        if col2.button("Save info"):
           params_upload(info_dict, info_name)
        st.session_state['model'] = model

    if state == "DataFrame":
        st.header("데이터프레임 하이퍼파라미터 설정")
        dataframe_set()

    elif state == "Sound":
        st.header("사운드 하이퍼파라미터 설정")
        sound_set()
    
    elif state == "Image":
        st.header("이미지 하이퍼파라미터 설정")
        image_set()



def trian_data():
    """
    데이터 학습 메뉴.
    """
    with st.sidebar:
        val = stx.stepper_bar(steps=["길라잡이", "데이터 업로드",  "하이퍼파라미터 설정", "모델 학습"], is_vertical=True, lock_sequence=False)
    train_data_menu(val)

def model_training(data:str, params:str):
    """
    모델 학습.
    """
    st.header("모델 학습 - {}".format(st.session_state['params']))
    model = copy.deepcopy(st.session_state['params'])
    col1, col2,col3,col4,col5 = st.columns(5, gap = "large")
    train = col1.button("학습하기")

    col1, col2 = st.columns(2, gap = "large")
    remain_time = st.empty()
    message = st.empty()
    bar = st.empty()
    Empty = st.empty()
    if train:
        start = 1
        try:
            requests.post(SERVER + '/hyun/prepare_run_model', params = {'filename': st.session_state['params']})
            requests.post(SERVER + '/hyun/run_model', params = {'filename':st.session_state['params'], 'model':st.session_state.DataType}, timeout=0.000001)
        except requests.exceptions.ReadTimeout:
            pass

        history = pd.DataFrame(columns=["epoch","train_loss","train_accuracy","validation_loss","validation_accuracy","timestamp"])
        bar = st.progress(0)
       
        
        remain_time.write("")

        while start == 1:
            time.sleep(0.1)
            try:
                response = requests.get(SERVER + '/hyun/processlog', params = {'filename':st.session_state['params']})
                df_temp = pd.DataFrame(json.loads(response.json()))
                if len(df_temp) >= 0:
                    message.info("학습 중...")
                    if len(df_temp) > 0:
                        if df_temp['timestamp'].min() == 0:
                            message.success("학습 완료.")
                            bar.progress(100)
                            start = 0
                            break
                        else:
                            history = pd.concat([history, df_temp], axis=0)
                            history["epoch"] = history["epoch"].astype(int)
                            now_epoch = history["epoch"].max()
                            history.drop_duplicates(subset=['epoch'], keep='last', inplace=True)
                            if st.session_state.model == "LSTM":
                                #drop column "accuracy"
                                Empty.table(history.drop(columns=["train_accuracy", "validation_accuracy"]))
                            elif st.session_state.model == "IForest":
                                pass
                            else:
                                Empty.table(history)
                            timestamp = history["timestamp"].values
                            now = timestamp[len(timestamp)-1]
                            now_1 = timestamp[len(timestamp)-2]
                            now = datetime.datetime.fromtimestamp(now)
                            now_1 = datetime.datetime.fromtimestamp(now_1)
                            remain = (now - now_1) * (st.session_state['epochs'] - now_epoch)
                            remain_time.info("예상 남은 시간 : {}".format(remain))
                            bar.progress(now_epoch/st.session_state['epochs'])
                            
            #json typeerror
            except TypeError:
                message.error("전처리 중...")
                continue
            except:
                continue
        if len(history) > 3:
            history.set_index('epoch', inplace=True)
            if st.session_state.model == "LSTM":
                #drop column "accuracy"
                st.line_chart(history.drop(columns=["train_accuracy", "validation_accuracy", "timestamp"]))
            elif st.session_state.model == "IForest":
                pass
            else:
                st.line_chart(history[['train_loss', 'validation_loss', 'train_accuracy', 'validation_accuracy']])
            remain_time.write("")
     
     
    




            
        
        
        




def test_data(state:str = None):
    page_predict()
    
        

def preprocessing():
    """
    데이터 전처리 메뉴.
    """
    with st.sidebar:
        val = stx.stepper_bar(steps=["길라잡이", "데이터 살펴보기",  "데이터 열 설정", "고급 설정"], is_vertical=True, lock_sequence=False)
    preprocessing_menu(val)

def preprocessing_menu(step:int):
    """
    데이터 전처리 메뉴.
    """
    if step == 0:
        html_view(PAGES_DIR+"data preprocessing.html", 1800)

        pass

    elif step == 1:
        st.header("데이터 살펴보기")
        uploaded_file = st.file_uploader("Choose a file", accept_multiple_files=False)
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file, encoding='utf-8')
            st.write(df)
        
            #draw chart
            #select x axis
            x_axis = st.selectbox("Select x axis", df.columns)
            #select y axis
            y_axis = st.multiselect("Select y axis", df.columns)
            fig = go.Figure()
            if x_axis is not None:
                if y_axis is not None:
                    for i in y_axis:
                        fig.add_trace(go.Scatter(x=df[x_axis], y=df[i], mode='lines', name=i))
                    fig.update_layout(autosize=False, width=1300, height=500,)
                    st.subheader("Chart")
                    st.plotly_chart(fig)

            fig = px.imshow(df.corr(), color_continuous_scale='greys', text_auto=True)
            fig.update_layout(autosize=False, width=1300, height=1000,)
            
            st.subheader("피어슨 상관계수 히트맵")
            try:
                recommend = []
                for i in range(len(df.corr())):
                    recommend.append(len(df.corr().iloc[i][df.corr().iloc[i].abs() < 0.05]))
                recommend = pd.DataFrame(recommend, index=df.columns, columns=["recommend"])
                recommend = recommend.sort_values(by="recommend", ascending=False)
                if len(recommend) > 6:
                    st.write("Recommend columns to drop. 1st : {}, 2nd : {}, 3rd : {}".format(recommend.index[1], recommend.index[2], recommend.index[3]))
                elif len(recommend) > 5:
                    st.write("Recommend columns to drop. 1st : {}, 2nd : {}".format(recommend.index[1], recommend.index[2]))
                elif len(recommend) > 4:
                    st.write("Recommend columns to drop. 1st : {}".format(recommend.index[1]))
                
                st.write("This is for reference only. It does not guarantee a sufficient effect.")
            except:
                pass    
            st.plotly_chart(fig)

    elif step == 2:
        st.header("데이터 열 설정")
        st.subheader("열 선택")
        uploaded_file = st.file_uploader("Choose a file", accept_multiple_files=False)
        if uploaded_file is not None:
            
            df = pd.read_csv(uploaded_file)
            st.write(df)
        
            
            col1, col2 = st.columns(2)
            drop_list = col1.multiselect("삭제할 열 선택.", df.columns)
            col2.markdown("""
            """)
            col2.markdown("""
            """)
            if col2.button("삭제"):
                try:
                    df = df.drop(drop_list, axis=1)
                    col1.markdown("""
                    열 삭제 완료
                    """)
                    col1.write(df)
                    st.session_state['df'] = df
                    btn = col2.download_button(
                        label="Download CSV file",
                        data=df.to_csv(index=False),
                        file_name="{}_preprocessed.csv".format(uploaded_file.name.split(".")[0]),
                        mime="df/csv"
                    )
                except:
                    st.write("삭제 오류")

            st.subheader("시간, 레이블 열 설정 ")
            col1, col2, col3, col4 = st.columns(4)
            init_time= col1.selectbox("시간 열 선택.", df.columns)
            init_Label = col2.selectbox("레이블 열 선택.", df.columns)
            col3.markdown("""""")
            col3.markdown("""""")
        
            if col3.button("설정"):
                try:
                    df.rename(columns={init_time:"Time", init_Label:"label"}, inplace=True)
                    col4.markdown("""
                    설정 완료.
                    """)
                    col4.write(df.columns)
                    st.session_state['df'] = df
                    btn = col3.download_button(
                        label="Download CSV file",
                        data=df.to_csv(index=False),
                        file_name="{}_preprocessed.csv".format(uploaded_file.name.split(".")[0]),
                        mime="df/csv"
                    )
                except:
                    st.write("설정 오류.")
            

    elif step == 3:
        st.header("고급 설정")
        uploaded_file = st.file_uploader("Choose a file", accept_multiple_files=False)
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            startup(data_id="1", data=df)
            components.iframe("/dtale/main/1", height=800) 


def realtime_detection():
    """
    실시간 이상탐지 메뉴.
    """
    with st.sidebar:
        page_guide_realtime()
        val = stx.stepper_bar(steps=["모델 선택", "데이터 컬럼 설정", "실시간 분석"], is_vertical=True, lock_sequence=True)
    if val == 0:
        realtime_datapreview()
    elif val == 1:
        realtime_columnsetting()
    elif val == 2:
        realtime_analysis()

def realtime_columnsetting():
    st.subheader("센서 데이터 열 설정")
    col1, col2, col3 = st.columns(3)
    select_time_col = col1.selectbox("시간 열 선택.", st.session_state['realtime_cols'])
    select_feature_col = col2.multiselect("속성 열 선택.", st.session_state['realtime_cols'])
    col3.markdown("""""")
    col3.markdown("""""")

    savebutton=col3.button("설정")
    df = pd.DataFrame(columns=[select_time_col]+select_feature_col)
    st.write("선택된 열 미리보기.")
    st.write(df)
    df.set_index(select_time_col, inplace=True)
    
    if savebutton:
        st.session_state['realtime_df'] = df
        st.success("설정 완료.")
    
def realtime_analysis():
    st.subheader("실시간 분석")
    st.write("실시간 분석을 시작합니다.")
    st.write("분석을 종료하려면 '분석 종료' 버튼을 누르세요.")
    df = st.session_state['realtime_df']
    col1, col2 = st.columns(2)
    realtime_start = col1.button("분석 시작")
    realtime_stop = col2.button("분석 종료")
    data_preview = st.empty()
    mama = st.empty()
    chart = st.empty()
    col_list = st.session_state['realtime_df'].columns
    
    if realtime_start:
        st.session_state['realtime_start'] = True
        st.session_state['realtime_stop'] = False
        while st.session_state['realtime_start']:
            if st.session_state['realtime_stop']:  
                break
            time.sleep(0.01)
            response = requests.get(SERVER + '/realtime/getdata')
            df_temp = pd.DataFrame(json.loads(response.json()))
            df_temp.set_index(df.index.name, inplace=True)
            

            #df의 컬럼과 같게 drop한다.
            df_temp = df_temp[col_list]
            data_preview.write(df_temp.tail(1))
            df = df.append(df_temp)
            df[col_list] = df[col_list].astype(float)
            if len(df) >100:
                df = df.tail(100)
                chart.line_chart(df, width = 900, height = 700)
            if abs(float(json.loads(response.json())[0]['accelerometerAccelerationY']) - 0.5) > 2.5 or abs(float(json.loads(response.json())[0]['accelerometerAccelerationX']) - 0.5) > 2.5 or abs(float(json.loads(response.json())[0]['accelerometerAccelerationZ']) - 0.5) > 2.5:
                mama.error("이상 발생!!")
                time.sleep(1)
            else:
                mama.success("정상")
                
            
                

    if realtime_stop:
        st.session_state['realtime_start'] = False
        st.session_state['realtime_stop'] = True
        requests.post(SERVER + '/hyun/stop_realtime')
    
            
    
def realtime_datapreview():
    """
    실시간 이상탐지 설정 메뉴.
    """
    st.header("모델 선택 및 설정")
    response = requests.get(SERVER + '/hyun/model_list')
    model_list = copy.deepcopy(response.json())
    try:
        model_select = st.session_state['params'] + ".h5"
        if model_select in model_list:
            index = model_list.index(model_select)
            temp = model_list[0]
            model_list[0] = model_list[index]
            model_list[index] = temp
        else:
            model_select = model_select[:-3] + ".pkl"
            index = model_list.index(model_select)
            temp = model_list[0]
            model_list[0] = model_list[index]
            model_list[index] = temp
    except:
        pass
    model_select = st.selectbox("", model_list)
    model_selectbutton = st.button("선택")
    
    st.write("") 
    col1, col2= st.columns(2)
    time_col = col1.empty()
    if model_selectbutton:
        response_1 = requests.get(SERVER + '/hyun/init_realtime', params={"model":model_select})
        st.session_state.params = model_select
        col1.subheader("선택된 모델 : {}".format(st.session_state.params))
        col1.write("학습 데이터의 정보")
        col1.write(pd.DataFrame(json.loads(response_1.json())).set_index("Unnamed: 0"))
        st.write("이제 서버에서 데이터를 수집합니다.")
        alert = st.empty()
        while True:
            alert.warning("데이터 수신 중...")
            time.sleep(1)
            response = requests.get(SERVER + '/realtime/getdata')
            if len(response.json()) > 20:
                alert.success("데이터 수신 완료.")
                break



        col2.write("현재 데이터의 Column은 다음과 같습니다.")
        col2.write(json.loads(response.json())[0])
        col2.error("주의 : 모델을 학습시킬 때의 속성과 다르면 오류가 발생합니다.")
        st.session_state['realtime_cols'] = json.loads(response.json())[0].keys()


    #여기 수정
       

        
        
    

        




            
            

            



        
        

    




    
    
    

if __name__ == "__main__":
    set_config()
    menu = set_sidebar()
    


    if menu == "데이터 학습":
        trian_data()
    elif menu == "데이터 예측":
        page_predict()
    elif menu == "데이터 전처리":
        preprocessing()
    elif menu == "실시간 이상탐지":
        realtime_detection()
    elif menu == "초기화":
        response = requests.post(SERVER + '/init')
        st.legacy_caching.clear_cache()
        menu = "데이터 학습"
        st.write("캐시가 초기화 되었습니다.")
        st.write("메뉴를 다시 선택해주세요.")
        
        
        
        
    
   