#	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
#	
#	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)


generate_regions_full_prototype() {
	local dimensions=$1
	local type=$2
	local commaatend
	
	/bin/echo '/** \brief Register a task '${type}' access on a '${dimensions}'-dimensional region of addresses */'
	/bin/echo 'void nanos6_register_region_'${type}'_depinfo'${dimensions}'('
	
	if [ "${type}" = "reduction" ] || [ "${type}" = "weak_reduction" ] ; then
		/bin/echo "${indentation}	int reduction_operation, int reduction_index,"
	fi
	
	/bin/echo '	void *handler /** Task handler */,'
	/bin/echo '	int symbol_index /** Argument identifier */,'
	/bin/echo '	char const *region_text /** Stringified contents of the dependency clause */,'
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
	
	if [ $(echo ${name} | sed 's/reduction//g') != ${name} ] ; then
		/bin/echo "${indentation}	int reduction_operation, int reduction_index,"
	fi
	
	/bin/echo "${indentation}	void *handler /** Task handler */,"
	/bin/echo "${indentation}	int symbol_index /** Argument identifier */,"
	/bin/echo "${indentation}	char const *region_text /** Stringified contents of the dependency clause */,"
	/bin/echo "${indentation}	void *base_address,"
	/bin/echo "${indentation}	/* First is the continuous dimension in bytes, the rest are based on the previous dimension */"
	/bin/echo "${indentation}	/* dimXstart is the first index/byte and dimXend is the next byte/index outside of the region */"
	for currentdimension in $(seq 1 ${dimensions}) ; do
		if [ ${currentdimension} -eq ${dimensions} ] ; then
			commaatend=""
		else
			commaatend=","
		fi
		
		/bin/echo "${indentation}	long dim${currentdimension}size, long dim${currentdimension}start, long dim${currentdimension}end${commaatend}"
	done
	/bin/echo -n "${indentation})"
}


generate_release_full_prototype() {
	local dimensions=$1
	local type=$2
	local commaatend
	
	/bin/echo '/** \brief Inform that the rest of the task code will no longer perform any '${type}' operation over a '${dimensions}'-dimensional region of addresses */'
	/bin/echo 'void nanos6_release_'${type}'_'${dimensions}'('
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


generate_release_named_prototype() {
	local dimensions=$1
	local name=$2
	local indentation=$3
	local commaatend
	
	if [ "${indentation}" != "" ] ; then
		indentation=$(emit_tabs ${indentation})
	fi
	
	/bin/echo "void ${name}("
	/bin/echo "${indentation}	void *base_address,"
	/bin/echo "${indentation}	/* First is the continuous dimension in bytes, the rest are based on the previous dimension */"
	/bin/echo "${indentation}	/* dimXstart is the first index/byte and dimXend is the next byte/index outside of the region */"
	for currentdimension in $(seq 1 ${dimensions}) ; do
		if [ ${currentdimension} -eq ${dimensions} ] ; then
			commaatend=""
		else
			commaatend=","
		fi
		
		/bin/echo "${indentation}	long dim${currentdimension}size, long dim${currentdimension}start, long dim${currentdimension}end${commaatend}"
	done
	/bin/echo -n "${indentation})"
}




generate_regions_api_type() {
	local dimensions=$1
	local name=$2
	
	/bin/echo -n "void ${name}("
	
	if [ x$(echo "${name}" | sed 's/reduction//g') != x${name} ] ; then
		/bin/echo -n "int, int, "
	fi
	
	/bin/echo -n "void *, int, char const*, void *"
	for currentdimension in $(seq 1 ${dimensions}) ; do
		/bin/echo -n ", long, long, long"
	done
	/bin/echo -n ")"
}

generate_release_api_type() {
	local dimensions=$1
	local name=$2
	
	/bin/echo -n "void ${name}("
	/bin/echo -n "void *"
	for currentdimension in $(seq 1 ${dimensions}) ; do
		/bin/echo -n ", long, long, long"
	done
	/bin/echo -n ")"
}


generate_regions_parameter_list() {
	local dimensions=$1
	local name=$2
	
	if [ x$(echo "${name}" | sed 's/reduction//g') != x"${name}" ] ; then
		/bin/echo -n "reduction_operation, reduction_index, "
	fi
	
	/bin/echo -n 'handler, symbol_index, region_text, base_address'
	for currentdimension in $(seq 1 ${dimensions}) ; do
		/bin/echo -n ", dim${currentdimension}size, dim${currentdimension}start, dim${currentdimension}end"
	done
}


generate_release_parameter_list() {
	local dimensions=$1
	
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



