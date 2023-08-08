FROM tensorflow/tensorflow:latest-gpu
RUN pip install pandas \
    pip install numpy \
    pip install sklearn \
    pip install silence_tensorflow \
    pip install matplotlib \
    pip install split_folders \
    pip install ipython \
    pip install pyod\
    pip install opencv-contrib-python \
    pip install librosa 
RUN apt-get update
RUN apt-get install libsndfile1 -y 
RUN apt-get install libgl1-mesa-glx -y
CMD ["test.py"]
ENTRYPOINT ["python3"]