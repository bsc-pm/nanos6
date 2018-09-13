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
	for dimensions in $(seq 1 ${maxdimensions}) ; do
		name=nanos6_register_region_${type}_depinfo${dimensions}
		
		generate_regions_full_prototype ${dimensions} ${type}
		echo " {"
		
		echo -n "	typedef "
		generate_regions_api_type ${dimensions} ${name}_t
		echo ";"
		echo "	"
		
		echo "	static ${name}_t *symbol = NULL;"
		echo "	if (__builtin_expect(symbol == NULL, 0)) {"
		echo "		symbol = (${name}_t *) _nanos6_resolve_symbol_with_local_fallback("
		echo "			\"${name}\", \"multidimensional dependency\", NULL, \"NULL\""
		echo "		);"
		echo "	}"
		echo "	"
		
		echo -n "	(*symbol)("
		generate_regions_parameter_list ${dimensions} ${type}
		echo ");"
		
		echo "}"
		echo
	done
	echo
done

echo ''
echo ''
echo '#pragma GCC visibility pop'
