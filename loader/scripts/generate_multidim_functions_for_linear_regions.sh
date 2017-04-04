#!/bin/sh


if [ $# -lt 2 ] ; then
	echo Usage: $0 '<max dimensions>' '<type1> [type2 [type3 [...]]]' 1>&2
	exit 1
fi


maxdimensions=$1
shift


. $(dirname $0)/common.sh


echo '#include <nanos6/multidimensional-dependencies.h>'
echo '#include <nanos6/dependencies.h>'
echo ''
echo '#include "MultidimensionalAPI.hpp"'
echo ''
echo ''



for type in $* ; do
	case ${type} in
		read)
			registration_function="register_data_access<READ_ACCESS_TYPE, false>"
		;;
		write)
			registration_function="register_data_access<WRITE_ACCESS_TYPE, false>"
		;;
		readwrite)
			registration_function="register_data_access<READWRITE_ACCESS_TYPE, false>"
		;;
		weak_read)
			registration_function="register_data_access<READ_ACCESS_TYPE, true>"
		;;
		weak_write)
			registration_function="register_data_access<WRITE_ACCESS_TYPE, true>"
		;;
		weak_readwrite)
			registration_function="register_data_access<READWRITE_ACCESS_TYPE, true>"
		;;
		concurrent)
			registration_function="register_data_access<CONCURRENT_ACCESS_TYPE, false>"
		;;
		commutative)
			registration_function="register_data_access<COMMUTATIVE_ACCESS_TYPE, false>"
		;;
		*)
			echo "Warning: unimplemented access type ${type}." 1>&2
			continue
		;;
	esac
	
	for dimensions in $(seq 1 ${maxdimensions}) ; do
		generate_regions_named_prototype ${dimensions} "nanos_register_region_${type}_depinfo${dimensions}"
		echo " {"
		echo "	${registration_function}("
		echo "		handler, symbol_index, region_text, base_address,"
		
		for level in $(seq ${dimensions} -1 1) ; do
			if [ "${level}" -ne 1 ] ; then
				opt_comma=","
			else
				opt_comma=""
			fi
			
			echo "		dim${level}size, dim${level}start, dim${level}end${opt_comma}"
		done
		echo "	);"
		
		echo "}"
		echo
	done
	echo
done
