#!/bin/sh

#	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
#	
#	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)


if [ $# -lt 2 ] ; then
	echo Usage: $0 '<max dimensions>' '<type1> [type2 [type3 [...]]]' 1>&2
	exit 1
fi


maxdimensions=$1
shift


. $(dirname $0)/common.sh


echo '#include "symbol-resolver/resolve.h"'
# echo '#include "nanos6/multidimensional-dependencies.h"'
echo
echo
echo '#pragma GCC visibility push(default)'
echo
echo
for type in $* ; do
	if [ "${type}" = "reduction" ] || [ "${type}" = "weak_reduction" ] ; then
		continue
	fi
	
	for dimensions in $(seq 1 ${maxdimensions}) ; do
		echo "RESOLVE_API_FUNCTION_WITH_LOCAL_FALLBACK(nanos6_release_${type}_${dimensions}, \"multidimensional release\", NULL);"
	done
	echo
done
echo
echo
echo '#pragma GCC visibility pop'
