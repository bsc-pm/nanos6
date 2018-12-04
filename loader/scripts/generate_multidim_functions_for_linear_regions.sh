#!/bin/sh

#	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
#	
#	Copyright (C) 2015-2018 Barcelona Supercomputing Center (BSC)


if [ $# -lt 2 ] ; then
	echo Usage: $0 '<max dimensions>' '<type1> [type2 [type3 [...]]]' 1>&2
	exit 1
fi


maxdimensions=$1
shift


. $(dirname $0)/common.sh


echo '#include <nanos6/multidimensional-dependencies.h>'
echo ''
echo '#include "MultidimensionalAPI.hpp"'
echo '#include "Dependencies.hpp"'
echo ''
echo ''
echo '#pragma GCC visibility push(default)'
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
		weak_commutative)
			registration_function="register_data_access<COMMUTATIVE_ACCESS_TYPE, true>"
		;;
		reduction)
			registration_function="register_reduction_access<false>"
		;;
		weak_reduction)
			registration_function="register_reduction_access<true>"
		;;
		*)
			echo "Warning: unimplemented access type ${type}." 1>&2
			continue
		;;
	esac
	
	for dimensions in $(seq 1 ${maxdimensions}) ; do
		if [ ${dimensions} -eq 1 ] ; then
			if [ "${type}" = "reduction" ] || [ "${type}" = "weak_reduction" ] ; then
				# Reductions are already implemented using the multidimensional API for 1 dimension
				continue
			fi
		fi
		
		generate_regions_named_prototype ${dimensions} "nanos6_register_region_${type}_depinfo${dimensions}"
		echo " {"
		echo "	${registration_function}("
		
		if [ "${type}" = "reduction" ] || [ "${type}" = "weak_reduction" ] ; then
			echo "		reduction_operation, reduction_index,"
		fi
		
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

echo ''
echo ''
echo '#pragma GCC visibility pop'
