#!/bin/sh

#	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
#
#	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

export NANOS6_CONFIG="${DIR}/../scripts/nanos6.toml"
export NANOS6_CONFIG_OVERRIDE="a=b"

# Any test with "discrete" in the name uses the simpler discrete implementation
if test -z ${NANOS6_DEPENDENCIES} ; then
	if [[ "${*}" == *"discrete"* ]]; then
		export NANOS6_CONFIG_OVERRIDE="${NANOS6_CONFIG_OVERRIDE},loader.dependencies=discrete"
	fi
fi

if test -z ${NANOS6_SCHEDULING_POLICY} ; then
	if [[ "${*}" == *"fibonacci"* ]] || [[ "${*}" == *"task-for-nqueens"* ]] || [[ "${*}" == *"taskloop-nqueens"* ]] || [[ "${*}" == *"taskloopfor-nqueens"* ]]; then
		export NANOS6_CONFIG_OVERRIDE="${NANOS6_CONFIG_OVERRIDE},scheduler.policy=lifo"
	fi
fi

# Enable DLB for dlb-specific tests
if [[ "${*}" == *"dlb-"* ]]; then
	export NANOS6_CONFIG_OVERRIDE="${NANOS6_CONFIG_OVERRIDE},dlb.enable=true"
else
	export NANOS6_CONFIG_OVERRIDE="${NANOS6_CONFIG_OVERRIDE},dlb.enable=false"
fi

# If DLB is enabled clean the shared memory first
if [[ ${NANOS6_ENABLE_DLB} == "1" ]]; then
	# Only if the command is found
	if hash dlb_shm 2>/dev/null; then
		dlb_shm -d
	fi
fi

if test -z ${NANOS6} ; then
	if test "${*}" = "${*/.debug/}" ; then
		export NANOS6_CONFIG_OVERRIDE="${NANOS6_CONFIG_OVERRIDE},loader.variant=optimized"
		exec "${@}"
	else
		export NANOS6_CONFIG_OVERRIDE="${NANOS6_CONFIG_OVERRIDE},loader.variant=debug"
		# Regular execution
		"${@}"
	fi
else
	exec "${@}"
fi
