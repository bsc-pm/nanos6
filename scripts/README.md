# Package generation

Make a distributable and obtain the absolute path of the volume

```
cd $nanos6_builddir
make dist
nanos6_pkgdir=$(readlink -f .)
```

## Enterprise Linux (Centos, RHEL)

Go to nanos6/scripts, build and run the Docker image.

```
docker build -t centos_rpm_builder -f Dockerfile.centos .
docker run --rm -ti -v $nanos6_pkgdir:/tmp/nanos6_pkgdir -u $(id -u):$(id -g) centos_rpm_builder
```

Untar the distributable inside the container and build the rpm package.

```
tar -C /tmp -xf /tmp/nanos6_pkgdir/nanos6-*.tar.bz2
cd /tmp/nanos6-*
./configure --without-boost
make rpm RELEASE=1 BOOST=/opt/boost_1_65_1
cp scripts/RPMS/*.rpm /tmp/nanos6_pkgdir/
cp scripts/RPMS/x86_64/nanos6*.rpm /tmp/nanos6_pkgdir/
```

## Debian

Go to nanos6/scripts, build and run the Docker image.

```
docker build -t debian_deb_builder -f Dockerfile.debian .
docker run --rm -ti -v $nanos6_pkgdir:/tmp/nanos6_pkgdir -u $(id -u):$(id -g) debian_deb_builder
```

Untar the distributable inside the container and build the deb package.

```
tar -C /tmp -xf /tmp/nanos6_pkgdir/nanos6-*.tar.bz2
cd /tmp/nanos6-*
./configure --without-boost
make deb RELEASE=1 BOOST=/opt/boost_1_65_1
cp scripts/nanos6*.deb /tmp/nanos6_pkgdir/
```
