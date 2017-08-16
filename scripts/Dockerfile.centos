# Dockerfile
FROM centos:latest

RUN yum -y update && yum -y install \
        autoconf \
        automake \
        binutils-devel \
        bison \
        createrepo \
        expect \
        flex \
        gcc \
        gcc-c++ \
        gcc-gfortran \
        gperf \
        libtool \
        libxml2-devel \
        make \
        pkgconfig \
        rpm-build \
        rpm-sign \
        sqlite \
        sqlite-devel \
        texinfo \
        numactl-devel \
        hwloc-devel \
        papi-devel \
        elfutils-devel \
        wget \
    && yum clean all

RUN wget -qO- http://dl.bintray.com/boostorg/release/1.65.1/source/boost_1_65_1.tar.bz2 \
              | tar xj -C /opt
