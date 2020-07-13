#!/bin/sh -e
#	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
#
#	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)


generate_regions_full_prototype() {
	local dimensions=$1
	local type=$2
	local commaatend

	printf '%s\n' '/** \brief Register a task '${type}' access on a '${dimensions}'-dimensional region of addresses */'
	echo 'void nanos6_register_region_'${type}'_depinfo'${dimensions}'('

	if [ "${type}" = "reduction" ] || [ "${type}" = "weak_reduction" ] ; then
		echo "${indentation}	int reduction_operation, int reduction_index,"
	fi

	echo '	void *handler /** Task handler */,'
	echo '	int symbol_index /** Argument identifier */,'
	echo '	char const *region_text /** Stringified contents of the dependency clause */,'
	echo '	void *base_address,'
	echo '	/* First is the continuous dimension in bytes, the rest are based on the previous dimension */'
	echo '	/* dimXstart is the first index/byte and dimXend is the next byte/index outside of the region */'
	for currentdimension in $(seq 1 ${dimensions}) ; do
		if [ ${currentdimension} -eq ${dimensions} ] ; then
			commaatend=""
		else
			commaatend=","
		fi

		echo "	long dim${currentdimension}size, long dim${currentdimension}start, long dim${currentdimension}end${commaatend}"
	done
	printf '%s' ')'
}


generate_regions_named_prototype() {
	local dimensions=$1
	local name=$2
	local indentation=$3
	local commaatend

	if [ "${indentation}" != "" ] ; then
		indentation=$(emit_tabs ${indentation})
	fi

	echo "void ${name}("

	if [ $(echo ${name} | sed 's/reduction//g') != ${name} ] ; then
		echo "${indentation}	int reduction_operation, int reduction_index,"
	fi

	echo "${indentation}	void *handler /** Task handler */,"
	echo "${indentation}	int symbol_index /** Argument identifier */,"
	echo "${indentation}	char const *region_text /** Stringified contents of the dependency clause */,"
	echo "${indentation}	void *base_address,"
	echo "${indentation}	/* First is the continuous dimension in bytes, the rest are based on the previous dimension */"
	echo "${indentation}	/* dimXstart is the first index/byte and dimXend is the next byte/index outside of the region */"
	for currentdimension in $(seq 1 ${dimensions}) ; do
		if [ ${currentdimension} -eq ${dimensions} ] ; then
			commaatend=""
		else
			commaatend=","
		fi

		echo "${indentation}	long dim${currentdimension}size, long dim${currentdimension}start, long dim${currentdimension}end${commaatend}"
	done
	printf '%s' "${indentation})"
}


generate_release_full_prototype() {
	local dimensions=$1
	local type=$2
	local commaatend

	printf '%s\n' '/** \brief Inform that the rest of the task code will no longer perform any '${type}' operation over a '${dimensions}'-dimensional region of addresses */'
	echo 'void nanos6_release_'${type}'_'${dimensions}'('
	echo '	void *base_address,'
	echo '	/* First is the continuous dimension in bytes, the rest are based on the previous dimension */'
	echo '	/* dimXstart is the first index/byte and dimXend is the next byte/index outside of the region */'
	for currentdimension in $(seq 1 ${dimensions}) ; do
		if [ ${currentdimension} -eq ${dimensions} ] ; then
			commaatend=""
		else
			commaatend=","
		fi

		echo "	long dim${currentdimension}size, long dim${currentdimension}start, long dim${currentdimension}end${commaatend}"
	done
	printf '%s' ')'
}


generate_release_named_prototype() {
	local dimensions=$1
	local name=$2
	local indentation=$3
	local commaatend

	if [ "${indentation}" != "" ] ; then
		indentation=$(emit_tabs ${indentation})
	fi

	echo "void ${name}("
	echo "${indentation}	void *base_address,"
	echo "${indentation}	/* First is the continuous dimension in bytes, the rest are based on the previous dimension */"
	echo "${indentation}	/* dimXstart is the first index/byte and dimXend is the next byte/index outside of the region */"
	for currentdimension in $(seq 1 ${dimensions}) ; do
		if [ ${currentdimension} -eq ${dimensions} ] ; then
			commaatend=""
		else
			commaatend=","
		fi

		echo "${indentation}	long dim${currentdimension}size, long dim${currentdimension}start, long dim${currentdimension}end${commaatend}"
	done
	printf '%s' "${indentation})"
}




generate_regions_api_type() {
	local dimensions=$1
	local name=$2

	printf '%s' "void ${name}("

	if [ x$(echo "${name}" | sed 's/reduction//g') != x${name} ] ; then
		printf '%s' "int, int, "
	fi

	printf '%s' "void *, int, char const*, void *"
	for currentdimension in $(seq 1 ${dimensions}) ; do
		printf '%s' ", long, long, long"
	done
	printf '%s' ")"
}

generate_release_api_type() {
	local dimensions=$1
	local name=$2

	printf '%s' "void ${name}("
	printf '%s' "void *"
	for currentdimension in $(seq 1 ${dimensions}) ; do
		printf '%s' ", long, long, long"
	done
	printf '%s' ")"
}


generate_regions_parameter_list() {
	local dimensions=$1
	local name=$2

	if [ x$(echo "${name}" | sed 's/reduction//g') != x"${name}" ] ; then
		printf '%s' "reduction_operation, reduction_index, "
	fi

	printf 'handler, symbol_index, region_text, base_address'
	for currentdimension in $(seq 1 ${dimensions}) ; do
		printf '%s' ", dim${currentdimension}size, dim${currentdimension}start, dim${currentdimension}end"
	done
}


generate_release_parameter_list() {
	local dimensions=$1

	printf 'base_address'
	for currentdimension in $(seq 1 ${dimensions}) ; do
		printf '%s' ", dim${currentdimension}size, dim${currentdimension}start, dim${currentdimension}end"
	done
}


emit_tabs() {
	printf "%*s" $1 "" | sed 's/ /\t/g'
}



