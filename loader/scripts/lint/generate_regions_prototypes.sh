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

echo '/*'
echo '	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.'
echo '	'
echo '	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)'
echo '*/'
echo
echo
echo '#ifndef NANOS6_LINT_MULTIDIMENSIONAL_ACCESSES_H'
echo '#define NANOS6_LINT_MULTIDIMENSIONAL_ACCESSES_H'
echo
echo '#pragma GCC visibility push(default)'
echo
echo
echo 'enum nanos6_lint_multidimensional_accesses_api_t { nanos6_lint_multidimensional_accesses_api = 1 };'
echo
echo
echo "#ifndef NANOS6_LINT_MAX_ACCESS_DIMENSIONS"
echo "#define NANOS6_LINT_MAX_ACCESS_DIMENSIONS ${maxdimensions}"
echo "#endif"
echo
echo
echo "#ifdef __cplusplus"
echo "extern \"C\" {"
echo "#endif"
echo
echo

for type in $* ; do
	for dimensions in $(seq 1 ${maxdimensions}) ; do
		generate_regions_full_prototype ${dimensions} ${type}
		echo ";"
		echo
	done
	echo
done

echo
echo "#ifdef __cplusplus"
echo "}"
echo "#endif"
echo
echo
echo '#endif /* NANOS6_LINT_MULTIDIMENSIONAL_ACCESSES_H */'

