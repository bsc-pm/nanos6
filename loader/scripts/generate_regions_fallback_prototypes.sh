#!/bin/sh


if [ $# -lt 2 ] ; then
	echo Usage: $0 '<max dimensions>' '<type1> [type2 [type3 [...]]]' 1>&2
	exit 1
fi


maxdimensions=$1
shift


. $(dirname $0)/common.sh


for type in $* ; do
	for dimensions in $(seq 1 ${maxdimensions}) ; do
		generate_regions_named_prototype ${dimensions} "nanos_register_region_${type}_depinfo${dimensions}_fallback"
		echo ";"
	done
	echo
done
