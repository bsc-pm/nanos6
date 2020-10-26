#!/bin/bash
#
#	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
#
#	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
#
# Generate config depending on enabled flags.

# Stop on error
set -e

# Remove a single config section
remove_section () {
	contents=$1
	section=$2

	# Remove anything between the __require_{section} and __!require_{section}
	echo "${contents}" | perl -0777 -pe 's/^__require_'"${section}"'(.*)^__!require_'"${section}"'\n//gsm'
}

# Remove dangling requires and newlines
remove_unused_requires () {
	contents=$1

	# Remove any __require_* or __!require_* and clean up the file removing extra newlines.
	# (\n{2,}) -> match any group of two or more newlines
	echo "${contents}" | perl -pe 's/^__(!*)require_(.*)(\n*)$//g' | perl -0777 -pe 's/(\n{2,})/\n\n/gs'
}

template_file=$1
shift

# This sections must be in order with the arguments of the script
possible_sections=(CUDA OPENACC CLUSTER DLB CTF GRAPH VERBOSE EXTRAE PAPI PQOS)
disabled_sections=()

# Read the arguments using the possible_sections order to check which sections
# are disabled and must be removed
for section in "${possible_sections[@]}"; do
	if [ "$1" != "1" ]; then
		disabled_sections+=(${section})
	fi
	shift
done

# Read configure template
file_contents=$(cat ${template_file})

# For each disabled section, call remove_section
for section in ${disabled_sections[@]}; do
	file_contents=$(remove_section "${file_contents}" "${section}")
done

# Clean up file
file_contents=$(remove_unused_requires "${file_contents}")

# Return the processed file
echo "${file_contents}"