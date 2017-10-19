#!/bin/sh


if [ $# -lt 2 ] ; then
	echo Usage: $0 '<max dimensions>' '<type1> [type2 [type3 [...]]]' 1>&2
	exit 1
fi


maxdimensions=$1
shift


. $(dirname $0)/common.sh


echo '#include "indirect-symbols/resolve.h"'
echo '#include "multidim-region-dependency-fallbacks.h"'
echo '#include "api/nanos6.h"'
echo
echo
echo '#pragma GCC visibility push(default)'
echo ''
echo ''

for type in $* ; do
	for dimensions in $(seq 1 ${maxdimensions}) ; do
		name=nanos_register_region_${type}_depinfo${dimensions}
		
		generate_regions_full_prototype ${dimensions} ${type}
		echo " {"
		
		echo -n "	typedef "
		generate_regions_api_type ${dimensions} ${name}_t
		echo ";"
		echo "	"
	
		if [ "${type}" = "reduction" ] ; then
			fallback_name=nanos_register_region_readwrite_depinfo${dimensions}_fallback
		else
			fallback_name=${name}_fallback
		fi
		
		echo "	static ${name}_t *symbol = NULL;"
		echo "	if (__builtin_expect(symbol == NULL, 0)) {"
		echo "		symbol = (${name}_t *) _nanos6_resolve_symbol_with_local_fallback("
		echo "			\"${name}\", \"multidimensional dependency\", ${fallback_name}, \"${fallback_name}\""
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
