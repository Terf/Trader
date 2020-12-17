#!/bin/bash

# xhost + 127.0.0.1
docker run --rm --env-file .env -e DISPLAY=host.docker.internal:0 -v $HOME:/root -v $PWD/methods/models:/var/keras -it terf/rh-trader trader.py
# docker run --rm --env-file .env -e DISPLAY=host.docker.internal:0 -v $HOME:/root -v $PWD/methods/models:/var/keras -it terf/rh-trader -c "import trader; trader.train_lstm()"
