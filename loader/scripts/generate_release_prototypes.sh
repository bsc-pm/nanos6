#!/bin/sh


if [ $# -lt 2 ] ; then
	echo Usage: $0 '<max dimensions>' '<type1> [type2 [type3 [...]]]' 1>&2
	exit 1
fi


maxdimensions=$1
shift


. $(dirname $0)/common.sh

echo '#ifndef NANOS6_MULTIDIMENSIONAL_RELEASE_H'
echo '#define NANOS6_MULTIDIMENSIONAL_RELEASE_H'
echo
echo 'enum nanos6_multidimensional_release_api_t { nanos6_multidimensional_release_api = 1 };'
echo
echo
echo "#ifndef NANOS6_MAX_DEPENDENCY_DIMENSIONS"
echo "#define NANOS6_MAX_DEPENDENCY_DIMENSIONS ${maxdimensions}"
echo "#endif"
echo
echo
echo "#ifdef __cplusplus"
echo "extern \"C\" {"
echo "#endif"
echo
echo

for type in $* ; do
	if [ "${type}" = "reduction" ] ; then
		continue
	fi
	
	for dimensions in $(seq 1 ${maxdimensions}) ; do
		generate_release_full_prototype ${dimensions} ${type}
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
echo '#endif /* NANOS6_MULTIDIMENSIONAL_RELEASE_H */'

