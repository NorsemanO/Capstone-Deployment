FROM ubuntu:latest
LABEL authors="Xi"

RUN mkdir -p /opt/model

WORKDIR /opt/model

RUN apt-get update && apt-get install -y python3 pip
#RUN apt-get update && apt-get install -y python3 pip python3-flask python3-numpy python3-joblib

ADD Fert_Predict.py /opt/model
ADD run-model.sh /opt/model
ADD model_fert_clf.pkl /opt/model
ADD requirements.txt /opt/model


RUN python3 -m pip install -r requirements.txt --break-system-packages

EXPOSE 80

ENTRYPOINT "./run-model.sh"
