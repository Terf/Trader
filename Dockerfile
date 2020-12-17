FROM python:3.8
WORKDIR /opt
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
ENTRYPOINT [ "python" ]
