FROM python:3.8-slim-buster

RUN apt-get update
RUN apt-get install -y build-essential cmake git unzip pkg-config libopenblas-dev liblapack-dev libhdf5-dev python3-dev python3-pip

RUN pip3 install pip
RUN pip3 install --upgrade pip
# RUN sudo pip3 install setuptools
ARG SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
RUN python3 -m pip install python-dateutil tensorflow nbformat scipy matplotlib 'h5py<3.0.0' graphviz pydot-ng opencv-python keras plotly sklearn csv-to-json azure-storage-blob azure-storage-file-share azure-storage-file-datalake azure-batch ciso8601 joblib silence_tensorflow fancyimpute pytz boto3 pyarrow
RUN python3 -m pip install pandas==1.4.4
# Opcional para parsear rasters
RUN apt-get install -y gdal-bin python3-gdal libeccodes-dev libgdal-dev

# azcopy
COPY azcopy /usr/bin/
RUN python3 -m pip install setuptools==57.5.0
RUN python3 -m pip install gdal==2.4.0

# más pips, pero después para aprovechar la caché
RUN python3 -m pip install azure-data-tables

RUN python3 -m pip install tweepy

RUN python3 -m pip install firebase_admin

RUN python3 -m pip install adlfs
RUN python3 -m pip install psycopg2
