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

	# Remove anything between the __require_{section} and
	# __!require_{section}, inclusive
	printf "%s\n" "${contents}" | \
		awk '/^__require_'"${section}"'$/ { skip=1; next }; \
			/^__!require_'"${section}"'$/{skip=0; next}; \
			!skip{print}'
}

# Remove dangling requires
remove_unused_requires () {
	contents=$1

	# Remove any __require_* or __!require_* leftover lines
	printf "%s\n" "${contents}" | sed '/^__!\?require_.*$/d'
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
printf "%s\n" "${file_contents}"
