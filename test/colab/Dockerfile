FROM --platform=linux/amd64 python:3.7.13-slim-buster

RUN apt-get update \
    && apt-get install --no-install-recommends --assume-yes \
    libgdal-dev r-base build-essential libglib2.0-0 libglib2.0-dev \
    libsndfile1 libsndfile1-dev cmake libgcc1 gcc-7-base gcc-8-base \
    g++ g++-7 gcc python-apt libgirepository1.0-dev \
    libcairo2-dev libjpeg-dev libgif-dev python3-cairo-dev python3-dev \
    libbz2-dev libreadline-dev libssl-dev zlib1g-dev libsqlite3-dev wget \
    curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev \
    python-apt-common python3-apt python3-distutils-extra python-apt \
    vim git \
    && curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python3 - -y \
    && apt-get autoremove -y

ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

COPY build/colab-test/requirements-fixed.txt /tmp/requirements-fixed.txt
RUN --mount=type=cache,mode=0755,target=/root/.cache/pip pip install --no-dependencies -r /tmp/requirements-fixed.txt
WORKDIR /usr/src/app

CMD ["./test/colab/test.sh"]