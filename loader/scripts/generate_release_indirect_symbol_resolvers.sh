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


echo '#include "indirect-symbols/resolve.h"'
echo '#include "api/nanos6.h"'
echo
echo
echo '#pragma GCC visibility push(default)'
echo ''
echo ''
for type in $* ; do
	if [ "${type}" = "reduction" ] || [ "${type}" = "weak_reduction" ] ; then
		continue
	fi
	
	for dimensions in $(seq 1 ${maxdimensions}) ; do
		name=nanos6_release_${type}_${dimensions}
		
		generate_release_full_prototype ${dimensions} ${type}
		echo " {"
		
		echo -n "	typedef "
		generate_release_api_type ${dimensions} ${name}_t
		echo ";"
		echo "	"
		
		echo "	static ${name}_t *symbol = NULL;"
		echo "	if (__builtin_expect(symbol == NULL, 0)) {"
		echo "		symbol = (${name}_t *) _nanos6_resolve_symbol_with_local_fallback("
		echo "			\"${name}\", \"multidimensional release\", NULL, \"NULL\""
		echo "		);"
		echo "	}"
		echo "	"
		
		echo -n "	(*symbol)("
		generate_release_parameter_list ${dimensions}
		echo ");"
		
		echo "}"
		echo
	done
	echo
done

echo ''
echo ''
echo '#pragma GCC visibility pop'
