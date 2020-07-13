#!/bin/sh -e
#	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
#
#	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)

generate_regions_full_prototype() {
	local dimensions=$1
	local type=$2
	local commaatend

	printf '/** \\brief Simulate a task '${type}' access on a '${dimensions}'-dimensional region of addresses */\n'
	echo 'void nanos6_lint_register_region_'${type}'_'${dimensions}'('

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
	printf ')'
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

	echo "${indentation}	__attribute__((unused)) void *base_address,"
	echo "${indentation}	/* First is the continuous dimension in bytes, the rest are based on the previous dimension */"
	echo "${indentation}	/* dimXstart is the first index/byte and dimXend is the next byte/index outside of the region */"
	for currentdimension in $(seq 1 ${dimensions}) ; do
		if [ ${currentdimension} -eq ${dimensions} ] ; then
			commaatend=""
		else
			commaatend=","
		fi

		echo "${indentation}	__attribute__((unused)) long dim${currentdimension}size, __attribute__((unused)) long dim${currentdimension}start, __attribute__((unused)) long dim${currentdimension}end${commaatend}"
	done
	printf "${indentation})"
}

generate_regions_api_type() {
	local dimensions=$1
	local name=$2

	printf "void ${name}(void *"
	for currentdimension in $(seq 1 ${dimensions}) ; do
		printf ", long, long, long"
	done
	printf ")"
}

generate_regions_parameter_list() {
	local dimensions=$1
	local name=$2

	printf 'base_address'
	for currentdimension in $(seq 1 ${dimensions}) ; do
		printf ", dim${currentdimension}size, dim${currentdimension}start, dim${currentdimension}end"
	done
}

emit_tabs() {
	printf "%*s" $1 "" | sed 's/ /\t/g'
}
