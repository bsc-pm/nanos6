#!/bin/sh


if [ $# -lt 2 ] ; then
	echo Usage: $0 '<max dimensions>' '<type1> [type2 [type3 [...]]]' 1>&2
	exit 1
fi


maxdimensions=$1
shift


. scripts/generate_regions_prototype.sh


for type in $* ; do
	for dimensions in $(seq 1 ${maxdimensions}) ; do
		generate_regions_prototype ${dimensions} ${type}
		echo ';'
		echo
	done
	echo
done
