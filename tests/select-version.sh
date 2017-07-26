#!/bin/sh

#	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
#	
#	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)


if test -z ${NANOS6} ; then
	if test "${*}" = "${*/.debug/}" ; then
		export NANOS6=optimized
	else
		export NANOS6=debug
	fi
fi

exec "${@}"
