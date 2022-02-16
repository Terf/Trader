FROM python:3.6
WORKDIR /opt
COPY requirements.txt .
RUN pip install -r requirements.txt
# copy in different steps to cache
COPY methods/models ./methods/models
COPY methods/*.py ./methods/
COPY trader.py ./
ENTRYPOINT [ "python" ]
