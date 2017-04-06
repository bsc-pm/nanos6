#!/bin/sh


if [ $# -lt 2 ] ; then
	echo Usage: $0 '<max dimensions>' '<type1> [type2 [type3 [...]]]' 1>&2
	exit 1
fi


maxdimensions=$1
shift


. $(dirname $0)/common.sh


echo '#include "symbol-resolver/resolve.h"'
echo '#include "multidim-region-dependency-fallbacks.h"'
# echo '#include "nanos6/multidimensional-dependencies.h"'
echo
echo
for type in $* ; do
	for dimensions in $(seq 1 ${maxdimensions}) ; do
		if [ "${type}" = "reduction" ] ; then
			fallback_name=nanos_register_region_readwrite_depinfo${dimensions}_fallback
		else
			fallback_name=nanos_register_region_${type}_depinfo${dimensions}_fallback
		fi
		
		echo "RESOLVE_API_FUNCTION_WITH_LOCAL_FALLBACK(nanos_register_region_${type}_depinfo${dimensions}, \"multidimensional dependency\", ${fallback_name});"
	done
	echo
done
