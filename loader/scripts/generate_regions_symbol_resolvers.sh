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
#echo '#include "nanos6/multidimensional-dependencies.h"'

echo
echo
for type in $* ; do
	for dimensions in $(seq 1 ${maxdimensions}) ; do
		if [ "${type}" = "commutative" ] ; then
			fallback_name=nanos6_register_region_readwrite_depinfo${dimensions}
		else
			fallback_name=NULL
		fi
	
		echo "RESOLVE_API_FUNCTION_WITH_LOCAL_FALLBACK(nanos6_register_region_${type}_depinfo${dimensions}, \"multidimensional dependency\", NULL);"
	done
	echo
done
