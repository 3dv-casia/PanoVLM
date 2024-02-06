# 基于 ubuntu 20 的基础镜像，并且安装 cmake 和 g++ 编译工具
FROM hdgigante/python-opencv:4.7.0-ubuntu
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install --no-install-recommends -y  \
    build-essential \
    cmake \
    libboost-all-dev \
    libcgal-dev \
    libpcl-dev \
    libceres-dev 
# 安装 pcl ceres-solver
# RUN apt-get install -y --no-install-recommends libcgal-dev libpcl-dev libceres-dev

RUN rm -rf /var/lib/apt/lists/*
