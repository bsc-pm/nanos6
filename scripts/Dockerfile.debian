# Dockerfile
FROM debian:latest

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        autoconf \
        automake \
        binutils-dev \
        bison \
        build-essential \
        ca-certificates \
        curl \
        debhelper \
        devscripts \
        equivs \
        expect \
        fakeroot \
        file \
        flex \
        gfortran \
        gperf \
        libiberty-dev \
        libltdl-dev \
        libsqlite3-dev \
        libtool \
        libxml2-dev \
        lsb-release \
        pkg-config \
        reprepro \
        sqlite3 \
        texinfo \
        numactl \
        libhwloc-dev \
        libpapi-dev \
        libelf-dev \
        libdw-dev \
        zlib1g-dev \
        liblzma-dev \
        wget \
    && rm -rf /var/lib/apt/lists/*

RUN wget -qO- http://dl.bintray.com/boostorg/release/1.65.1/source/boost_1_65_1.tar.bz2 \
              | tar xj -C /opt
