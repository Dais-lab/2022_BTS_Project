# 2022 BTS Project

## Abstract

Failure or abnormalities of facilities in the manufacturing process lead to product defects and production facility operations, which causes great losses to the manufacturer. Therefore, to prevent this, artificial intelligence-based research is being actively conducted to diagnose facilities and predict abnormalities. However, in large companies, data experts reside to perform the tasks and establish and develop a smart factory, but most companies often find it difficult to collect data, start with preprocessing, and detect abnormalities.

Therefore, it was intended to develop a test bed that can perform pretreatment, learning, and prediction of actual data obtained from the manufacturing industry on the platform, and furthermore, an open platform for anomaly detection that can detect abnormalities in real time by directly connecting with sensors. The platform is designed to enhance the platform's usability and accessibility from a non-professional perspective and to easily obtain results through deep learning models or transfer learning using other models even if users do not have basic knowledge including deep learning, artificial intelligence, and statistics.

## Contributors

### Members
`Changhyun Lee`|`Junhyeok Choi`|`Taeyeon Kim`|`Sunyoung Kim`

### Advisor
`Sudong Lee`

### Contribution
`Changhyun Lee` &nbsp;: Project Manager, Frontend, Backend, Arduino, Realtime anomaly detection    
`Junhyeok Choi` &nbsp;: Time series anomaly detection, Image anomaly detection, Sound anomaly detection    
`Taeyeon Kim`   &nbsp;: Backend, Arduino     
`Seonyeong Kim`  &nbsp;: Time series anomaly detection     

## Tools

environment : Ubuntu 20.04 - RTX 3090

- Python 3.8
- Tensorflow
- Docker
- Fastapi
- Streamlit
- Plotly
- Numpy
- Pandas
- Scikit - learn
- Matplotlib
- OpenCV
- librosa

### To Use


(도커에 대한 이해도가 낮을때 개발된 것이라 예상치못한 오류가 발생할 수 있습니다. 양해바랍니다.)
1. work directory 이동 - cd 2022BTS_DaisLab_ADP-main
2. Docker Image 제작.
    
    sudo docker build --tag test:0.1 ./
    
3. 종속성 설치
    
    python venv 생성
    
    1. python -m venv venv
    2. . venv/bin/activate
    3. pip install -r requirements.txt

4. config.py 수정
    1. Backend_Address : uvicorn backend:app —reload 명령어 실행 시 출력되는 주소.
    2. BASE_DIR : 2022BTS_DaisLab_ADP-main 폴더의 절대 경로
    3. SERVER_PASSWORD : 서버 사용자 패스워드
    
    수정 후 주석 처리된 것을 해제.
    
5. uvicorn backend:app --reload & dtale-streamlit run frontend.py

### 전체 디렉토리 구조. 

빈 디렉토리는 Git에 반영이 되지 않아서 디렉토리 추가 후 실행 필요.

<img width="532" alt="스크린샷 2023-01-05 오후 7 45 41" src="https://user-images.githubusercontent.com/74236661/210762183-1ed7ec15-f000-4cf8-b987-0b04527827fe.png">
