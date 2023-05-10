#!/bin/sh -e
#	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
#
#	Copyright (C) 2023 Barcelona Supercomputing Center (BSC)

# Version of hwloc. Change this to update hwloc.
hwloc_version=2.3.0

if ! command -v autoreconf; then
	echo "error: autoreconf not found"
	exit 1
fi

if [ ! -d "hwloc-${hwloc_version}" ]
then
  mkdir hwloc-${hwloc_version}
  tar -zxvf hwloc-${hwloc_version}.tar.gz --strip-components=1 -C hwloc-${hwloc_version}
fi

hwloc-${hwloc_version}/autogen.sh
autoreconf -fiv
