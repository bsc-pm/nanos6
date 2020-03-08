#	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
#
#	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)


generate_regions_full_prototype() {
	local dimensions=$1
	local type=$2
	local commaatend

	/bin/echo '/** \brief Simulate a task '${type}' access on a '${dimensions}'-dimensional region of addresses */'
	/bin/echo 'void nanos6_lint_register_region_'${type}'_'${dimensions}'('

	# if [ "${type}" = "reduction" ] || [ "${type}" = "weak_reduction" ] ; then
	# 	/bin/echo "${indentation}	int reduction_operation, int reduction_index,"
	# fi

	/bin/echo '	void *base_address,'
	/bin/echo '	/* First is the continuous dimension in bytes, the rest are based on the previous dimension */'
	/bin/echo '	/* dimXstart is the first index/byte and dimXend is the next byte/index outside of the region */'
	for currentdimension in $(seq 1 ${dimensions}) ; do
		if [ ${currentdimension} -eq ${dimensions} ] ; then
			commaatend=""
		else
			commaatend=","
		fi

		/bin/echo "	long dim${currentdimension}size, long dim${currentdimension}start, long dim${currentdimension}end${commaatend}"
	done
	/bin/echo -n ')'
}


generate_regions_named_prototype() {
	local dimensions=$1
	local name=$2
	local indentation=$3
	local commaatend

	if [ "${indentation}" != "" ] ; then
		indentation=$(emit_tabs ${indentation})
	fi

	/bin/echo "void ${name}("

	# if [ $(echo ${name} | sed 's/reduction//g') != ${name} ] ; then
	# 	/bin/echo "${indentation}	int reduction_operation, int reduction_index,"
	# fi

	/bin/echo "${indentation}	__attribute__((unused)) void *base_address,"
	/bin/echo "${indentation}	/* First is the continuous dimension in bytes, the rest are based on the previous dimension */"
	/bin/echo "${indentation}	/* dimXstart is the first index/byte and dimXend is the next byte/index outside of the region */"
	for currentdimension in $(seq 1 ${dimensions}) ; do
		if [ ${currentdimension} -eq ${dimensions} ] ; then
			commaatend=""
		else
			commaatend=","
		fi

		/bin/echo "${indentation}	__attribute__((unused)) long dim${currentdimension}size, __attribute__((unused)) long dim${currentdimension}start, __attribute__((unused)) long dim${currentdimension}end${commaatend}"
	done
	/bin/echo -n "${indentation})"
}




generate_regions_api_type() {
	local dimensions=$1
	local name=$2

	/bin/echo -n "void ${name}("

	# if [ x$(echo "${name}" | sed 's/reduction//g') != x${name} ] ; then
	# 	/bin/echo -n "int, int, "
	# fi

	/bin/echo -n "void *"
	for currentdimension in $(seq 1 ${dimensions}) ; do
		/bin/echo -n ", long, long, long"
	done
	/bin/echo -n ")"
}


generate_regions_parameter_list() {
	local dimensions=$1
	local name=$2

	# if [ x$(echo "${name}" | sed 's/reduction//g') != x"${name}" ] ; then
	# 	/bin/echo -n "reduction_operation, reduction_index, "
	# fi

	/bin/echo -n 'base_address'
	for currentdimension in $(seq 1 ${dimensions}) ; do
		/bin/echo -n ", dim${currentdimension}size, dim${currentdimension}start, dim${currentdimension}end"
	done
}


emit_tabs() {
	local n=$1

	if [ -z $1 ] ; then
		return
	fi

	while true ; do
		case $n in
			0)
				return
				;;
			1)
				/bin/echo -n "	"
				return
				;;
			2)
				/bin/echo -n "		"
				return
				;;
			3)
				/bin/echo -n "			"
				return
				;;
			4)
				/bin/echo -n "				"
				return
				;;
			*)
				/bin/echo -n "				"
				n=$(($n - 4))
				;;
		esac
	done
}



