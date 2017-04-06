#!/bin/sh


if [ $# -lt 2 ] ; then
	echo Usage: $0 '<max dimensions>' '<type1> [type2 [type3 [...]]]' 1>&2
	exit 1
fi


maxdimensions=$1
shift


. $(dirname $0)/common.sh


echo '#include "nanos6/multidimensional-release.h"'
echo
echo

for type in $* ; do
	if [ "${type}" = "reduction" ] ; then
		continue
	fi
	
	generate_release_named_prototype 1 "nanos_release_${type}_1_fallback"
	echo " {"
	echo "}"
	echo
	
	for dimensions in $(seq 2 ${maxdimensions}) ; do
		generate_release_named_prototype ${dimensions} "nanos_release_${type}_${dimensions}_fallback"
		echo " {"
		
		if [ ${dimensions} -eq 1 ] ; then
			echo "	char *as_array = base_address;"
		else
			echo -n "	char (*as_array)"
			for level in $(seq $((${dimensions} - 1)) -1 1) ; do
				echo -n "[dim${level}size]"
			done
			echo " = base_address;"
		fi
		echo "	"
		
		for level in $(seq ${dimensions} -1 2) ; do
			indentation=$((${dimensions} - ${level} + 1))
			
			emit_tabs ${indentation}
			echo "for (long index${level} = dim${level}start; index${level} < dim${level}end; index${level}++) {"
		done
		
		emit_tabs ${dimensions}
		echo -n "nanos_release_${type}_1(&as_array"
		for level in $(seq ${dimensions} -1 2) ; do
			echo -n "[index${level}]"
		done
		echo "[dim1start], dim1end - dim1start, 0, dim1end - dim1start);"
		
		for level in $(seq ${dimensions} -1 2) ; do
			emit_tabs $((${level} - 1))
			echo "}"
		done
		
		echo "}"
		echo
	done
	echo
done
