#!/bin/sh -e
#	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
#
#	Copyright (C) 2023 Barcelona Supercomputing Center (BSC)

# Folders where the dependency packages should be located
deps_folder=deps
hwloc_folder=${deps_folder}/hwloc

# The default hwloc version. This value should be changed when updating the default hwloc
hwloc_default_version=2.9.1

usage() {
	echo "Usage: $1 [--embed-hwloc VERSION] [--help]"
	echo ""
	echo "  Description:"
	echo "    The script extracts and embeds the external dependencies in the '${deps_folder}' folder, prepares"
	echo "    autotools for these packages, and then, runs the autoreconf for the current package. The"
	echo "    current default versions for the embedded dependencies are:"
	echo "      - hwloc: ${hwloc_default_version}"
	echo ""
	echo "  Options:"
	echo "    --embed-hwloc VERSION  Embed the hwloc sources in '${hwloc_folder}' with the '${hwloc_folder}-VERSION.tar.gz'"
	echo "                           tarball. The VERSION value can be 'default' to use the default hwloc version"
	echo "    --help, -h             Show the current help and exit"
}

error() {
	echo "==== ERROR: $@"
	exit 1
}

verbose() {
	echo "==== INFO: $@"
}

# Verify if the script is being is executed from the top level directory
if [ ! -f src/scheduling/Scheduler.hpp ]; then
	error "must execute $0 at the top level directory"
fi

if ! command -v tar ; then
	error "tar command not found"
fi

if ! command -v autoreconf ; then
	error "autoreconf command not found"
fi

# If not specified, use the default version distributed along with the runtime version
hwloc_version="default"

# Do not regenerate embedded dependencies by default
hwloc_regenerate=0

while [ $# -gt 0 ]; do
	if [ "$1" = "--embed-hwloc" ]; then
		if [ $# -lt 2 ]; then
			error "option --embed-hwloc requires the version number or 'default'"
		fi
		hwloc_version=$2
		hwloc_regenerate=1
		shift 2
	elif [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
		usage $0
		exit 0
	else
		usage $0
		error "option $1 not valid"
	fi
done

# Set the proper version if the user requested the default
if [ "${hwloc_version}" = "default" ] || [ "${hwloc_version}" = "DEFAULT" ]; then
	hwloc_version=${hwloc_default_version}
fi

hwloc_tarball=${hwloc_folder}-${hwloc_version}.tar.gz
if [ ! -f "${hwloc_tarball}" ]; then
	error "${hwloc_tarball} not found"
fi

rm -rf deps/jemalloc-5.3.0
mkdir deps/jemalloc-5.3.0
tar -zxvf deps/jemalloc-5.3.0.tar.gz --strip-components=1 -C deps/jemalloc-5.3.0
(cd deps/jemalloc-5.3.0 && ./autogen.sh)

if [ ! -d "${hwloc_folder}" ] || [ ${hwloc_regenerate} -eq 1 ]; then
	rm -rf ${hwloc_folder}
	mkdir ${hwloc_folder}
	tar -zxvf ${hwloc_tarball} --strip-components=1 -C ${hwloc_folder}
	verbose "successfully extracted '${hwloc_tarball}' in '${hwloc_folder}'"
else
	verbose "reusing previously extracted hwloc in '${hwloc_folder}'"
fi

if [ -f ${hwloc_folder}/autogen.sh ]; then
	verbose "generating autotools from '${hwloc_folder}' package"
	./${hwloc_folder}/autogen.sh
else
	verbose "autogen.sh script not found in '${hwloc_folder}'"
	verbose "skipping autotools generation from '${hwloc_folder}' package"
fi

if [ ! -f ${hwloc_folder}/configure ]; then
	error "${hwloc_folder}/configure not found"
fi

verbose "configuring autotools from current package"
autoreconf -fiv
