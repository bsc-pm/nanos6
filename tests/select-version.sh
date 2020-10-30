#!/bin/sh

#	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
#
#	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)

# The top build directory is passed on the first parameter
DIR=$1
shift

export NANOS6_CONFIG="${DIR}/scripts/nanos6.toml"

# Any test with "discrete" in the name uses the simpler discrete implementation
if [[ "${*}" == *"discrete"* ]]; then
	export NANOS6_CONFIG_OVERRIDE="${NANOS6_CONFIG_OVERRIDE},version.dependencies=discrete"
fi

if [[ "${*}" == *"fibonacci"* ]] || [[ "${*}" == *"task-for-nqueens"* ]] || [[ "${*}" == *"taskloop-nqueens"* ]] || [[ "${*}" == *"taskloopfor-nqueens"* ]]; then
	export NANOS6_CONFIG_OVERRIDE="${NANOS6_CONFIG_OVERRIDE},scheduler.policy=lifo"
fi

# Enable DLB for dlb-specific tests
if [[ "${*}" == *"dlb-"* ]]; then
	export NANOS6_CONFIG_OVERRIDE="${NANOS6_CONFIG_OVERRIDE},dlb.enabled=true"

	# If DLB is enabled clean the shared memory first
	if hash dlb_shm 2>/dev/null; then
		dlb_shm -d
	fi
else
	export NANOS6_CONFIG_OVERRIDE="${NANOS6_CONFIG_OVERRIDE},dlb.enabled=false"
fi

if test "${*}" = "${*/.debug/}" ; then
	export NANOS6_CONFIG_OVERRIDE="${NANOS6_CONFIG_OVERRIDE},version.debug=false"
	exec "${@}"
else
	export NANOS6_CONFIG_OVERRIDE="${NANOS6_CONFIG_OVERRIDE},version.debug=true"
	# Regular execution
	"${@}"
fi
