FROM python:3.12

ADD . /opt/ml_in_server2
WORKDIR /opt/ml_in_server2

# install packages by conda
RUN pip install -r requirements_prod.txt
CMD ["python", "server2.py"]
