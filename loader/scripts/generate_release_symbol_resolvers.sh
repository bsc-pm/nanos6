#!/bin/sh


if [ $# -lt 2 ] ; then
	echo Usage: $0 '<max dimensions>' '<type1> [type2 [type3 [...]]]' 1>&2
	exit 1
fi


maxdimensions=$1
shift


. $(dirname $0)/common.sh


echo '#include "symbol-resolver/resolve.h"'
echo '#include "multidim-release-fallbacks.h"'
# echo '#include "nanos6/multidimensional-dependencies.h"'
echo
echo
for type in $* ; do
	if [ "${type}" = "reduction" ] ; then
		continue
	fi
	
	for dimensions in $(seq 1 ${maxdimensions}) ; do
		echo "RESOLVE_API_FUNCTION_WITH_LOCAL_FALLBACK(nanos_release_${type}_${dimensions}, \"multidimensional release\", nanos_release_${type}_${dimensions}_fallback);"
	done
	echo
done
