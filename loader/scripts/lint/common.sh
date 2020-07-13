#!/bin/sh -e
#	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
#
#	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)

generate_regions_full_prototype() {
	local dimensions=$1
	local type=$2
	local commaatend

	printf '%s\n' '/** \brief Simulate a task '${type}' access on a '${dimensions}'-dimensional region of addresses */'
	printf '%s\n' 'void nanos6_lint_register_region_'${type}'_'${dimensions}'('

	printf '%s\n' '	void *base_address,'
	printf '%s\n' '	/* First is the continuous dimension in bytes, the rest are based on the previous dimension */'
	printf '%s\n' '	/* dimXstart is the first index/byte and dimXend is the next byte/index outside of the region */'
	for currentdimension in $(seq 1 ${dimensions}) ; do
		if [ ${currentdimension} -eq ${dimensions} ] ; then
			commaatend=""
		else
			commaatend=","
		fi

		printf '%s\n' "	long dim${currentdimension}size, long dim${currentdimension}start, long dim${currentdimension}end${commaatend}"
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

	printf '%s\n' "void ${name}("

	printf '%s\n' "${indentation}	__attribute__((unused)) void *base_address,"
	printf '%s\n' "${indentation}	/* First is the continuous dimension in bytes, the rest are based on the previous dimension */"
	printf '%s\n' "${indentation}	/* dimXstart is the first index/byte and dimXend is the next byte/index outside of the region */"
	for currentdimension in $(seq 1 ${dimensions}) ; do
		if [ ${currentdimension} -eq ${dimensions} ] ; then
			commaatend=""
		else
			commaatend=","
		fi

		printf '%s\n' "${indentation}	__attribute__((unused)) long dim${currentdimension}size, __attribute__((unused)) long dim${currentdimension}start, __attribute__((unused)) long dim${currentdimension}end${commaatend}"
	done
	printf '%s' "${indentation})"
}

generate_regions_api_type() {
	local dimensions=$1
	local name=$2

	printf '%s' "void ${name}(void *"
	for currentdimension in $(seq 1 ${dimensions}) ; do
		printf '%s' ", long, long, long"
	done
	printf '%s' ")"
}

generate_regions_parameter_list() {
	local dimensions=$1
	local name=$2

	printf '%s' 'base_address'
	for currentdimension in $(seq 1 ${dimensions}) ; do
		printf '%s' ", dim${currentdimension}size, dim${currentdimension}start, dim${currentdimension}end"
	done
}

emit_tabs() {
	printf "%*s" $1 "" | sed 's/ /\t/g'
}
