#!/bin/sh -e
#	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
#
#	Copyright (C) 2023 Barcelona Supercomputing Center (BSC)

# Folders where the dependency packages should be located
deps_directory=deps
hwloc_directory=${deps_directory}/hwloc
jemalloc_directory=${deps_directory}/jemalloc

# The default embedded dependency versions. This value should be changed when
# updating the default versions
hwloc_default_version=2.9.1
jemalloc_default_version=5.3.0

# Print the script usage and help
usage() {
	echo "Usage: $1 [--embed-hwloc VERSION] [--embed-jemalloc VERSION] [--help]"
	echo ""
	echo "  Prepare the configuration files for the Nanos6 package and its external"
	echo "  software dependencies. The script retrieves the external packages from"
	echo "  the '${deps_directory}' directory and extracts/embeds them there."
	echo ""
	echo "  By default, the script embeds the following versions:"
	echo "    hwloc: ${hwloc_default_version}"
	echo "    jemalloc: ${jemalloc_default_version}"
	echo ""
	echo "  Options:"
	echo "    --embed-hwloc VERSION     Embed the '${hwloc_directory}-VERSION.tar.gz'"
	echo "                              package as the external hwloc dependency instead"
	echo "                              of the default version. The VERSION value may be"
	echo "                              set to 'default' to use the default hwloc"
	echo "    --embed-jemalloc VERSION  Embed the '${jemalloc_directory}-VERSION.tar.gz'"
	echo "                              package as the external jemalloc dependency instead"
	echo "                              of the default version. The VERSION value may be"
	echo "                              set to 'default' to use the default jemalloc"
	echo "    --help, -h                Show the current help and exit"
}

# Print an error and abort execution
error() {
	echo "==== ERROR: $@"
	exit 1
}

# Print verbose information
verbose() {
	echo "==== INFO: $@"
}

# Return whether a file exist
exist_command() {
	command -v $1 >/dev/null 2>&1
	return $?
}

# Check whether a file exist and abort otherwise
check_file() {
	if [ ! -f $1 ]; then
		error "$1 not found"
	fi
}

# Check whether a command exists and abort otherwise
check_command() {
	if ! exist_command $1 ; then
		error "$1 command not found"
	fi
}

# Extract a package in a specific directory
extract_package() {
	local directory=$1
	local tarball=$2
	local regenerate=$3

	if [ ! -d "${directory}" ] || [ ${regenerate} -eq 1 ]; then
		rm -rf ${directory}
		mkdir ${directory}
		tar -zxf ${tarball} --strip-components=1 -C ${directory}
		verbose "successfully extracted '${tarball}' in '${directory}'"
	else
		verbose "reusing previously extracted '${directory}'"
	fi
}

# Prepare autotools for a package in a specific directory
autogen_package() {
	local directory=$1
	local autogen=$2

	currdir=$(pwd)
	cd ${directory}

	if exist_command $autogen ; then
		verbose "generating autotools from '${directory}' package"
		${autogen}
	else
		verbose "$autogen command or script not found in '${directory}'"
		verbose "skipping autotools generation from '${directory}' package"
	fi

	cd ${currdir}
}

# Verify whether the script is being is executed from the top level directory
if [ ! -f src/scheduling/Scheduler.hpp ]; then
	error "must execute $0 at the top level directory"
fi
currdir=$(pwd)

# If not specified, use the default version distributed along with the runtime version
hwloc_version="default"
jemalloc_version="default"

# Do not regenerate embedded dependencies by default
hwloc_regenerate=0
jemalloc_regenerate=0

while [ $# -gt 0 ]; do
	if [ "$1" = "--embed-hwloc" ]; then
		if [ $# -lt 2 ]; then
			error "option $1 requires the version number or 'default'"
		fi
		hwloc_version=$2
		hwloc_regenerate=1
		shift 2
	elif [ "$1" = "--embed-jemalloc" ]; then
		if [ $# -lt 2 ]; then
			error "option $1 requires the version number or 'default'"
		fi
		jemalloc_version=$2
		jemalloc_regenerate=1
		shift 2
	elif [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
		usage $0
		exit 0
	else
		usage $0
		error "option $1 not valid"
	fi
done

check_command "tar"
check_command "autoreconf"
check_command "autoconf"

# Set the proper version if the user requested the default
if [ "${hwloc_version}" = "default" ]; then
	hwloc_version=${hwloc_default_version}
fi
if [ "${jemalloc_version}" = "default" ]; then
	jemalloc_version=${jemalloc_default_version}
fi

hwloc_tarball=${hwloc_directory}-${hwloc_version}.tar.gz
jemalloc_tarball=${jemalloc_directory}-${jemalloc_version}.tar.gz

# Check that the package tarballs are available
check_file ${hwloc_tarball}
check_file ${jemalloc_tarball}

# Extract package tarballs if needed
extract_package ${hwloc_directory} ${hwloc_tarball} ${hwloc_regenerate}
extract_package ${jemalloc_directory} ${jemalloc_tarball} ${jemalloc_regenerate}

# Prepare autotools for each package
autogen_package ${hwloc_directory} "./autogen.sh"
autogen_package ${jemalloc_directory} "autoconf"

# Check their configure scripts exist
check_file "${hwloc_directory}/configure"
check_file "${jemalloc_directory}/configure"

# Prepare autotools for the Nanos6 package
verbose "configuring autotools from current package"
autoreconf -fiv
