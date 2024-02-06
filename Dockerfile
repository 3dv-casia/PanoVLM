FROM hdgigante/python-opencv:4.7.0-ubuntu
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install --no-install-recommends -y  \
    build-essential \
    cmake \
    libboost-all-dev \
    libcgal-dev \
    libpcl-dev \
    libceres-dev 

RUN rm -rf /var/lib/apt/lists/*
