#!/bin/sh

#	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
#	
#	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)

# Any test with "discrete" in the name will use the simpler discrete
# implementation.

if test -z ${NANOS6_DEPENDENCIES} ; then
  if "${*}" = "${*/discrete}" ; then
	export NANOS6_DEPENDENCIES=discrete
  fi
fi

if test -z ${NANOS6_SCHEDULING_POLICY} ; then
	if [[ "${*}" == *"fibonacci"* ]] || [[ "${*}" == *"task-for-nqueens"* ]]; then
		export NANOS6_SCHEDULING_POLICY=lifo
	fi
fi

if test -z ${NANOS6} ; then
	if test "${*}" = "${*/.debug/}" ; then
		export NANOS6=optimized
		exec "${@}"
	else
		export NANOS6=debug
		
		# Regular execution
		"${@}"
	fi
else
	exec "${@}"
fi

