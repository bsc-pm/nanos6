#!/bin/sh

#	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
#
#	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)


if [ $# -lt 2 ] ; then
	echo Usage: $0 '<max dimensions>' '<type1> [type2 [type3 [...]]]' 1>&2
	exit 1
fi


maxdimensions=$1
shift


. $(dirname $0)/common.sh


echo '#include <nanos6/lint-multidimensional-accesses.h>'
echo ''
echo ''
echo '#pragma GCC visibility push(default)'
echo ''
echo ''



for type in $* ; do

	for dimensions in $(seq 1 ${maxdimensions}) ; do
		# if [ ${dimensions} -eq 1 ] ; then
		# 	if [ "${type}" = "reduction" ] || [ "${type}" = "weak_reduction" ] ; then
		# 		# Reductions are already implemented using the multidimensional API for 1 dimension
		# 		continue
		# 	fi
		# fi

		generate_regions_named_prototype ${dimensions} "nanos6_lint_register_region_${type}_${dimensions}"
		echo " {"
		echo "	return;"
		echo "}"
		echo
	done
	echo
done

echo ''
echo ''
echo '#pragma GCC visibility pop'
