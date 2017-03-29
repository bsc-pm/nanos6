#!/bin/sh

if test -z ${NANOS6} ; then
	if test "${*}" = "${*/.debug/}" ; then
		export NANOS6=optimized
	else
		export NANOS6=debug
	fi
fi

exec "${@}"
