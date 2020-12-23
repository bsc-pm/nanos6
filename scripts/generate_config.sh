#!/bin/bash -e
#
#	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
#
#	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
#
# Generate config depending on enabled flags

# Stop on error
set -e

# Remove a single config section
remove_section () {
	file=$1
	section=$2

	# Remove anything between "__require_{section}" and "__!require_{section}"
	printf "%s\n" "${file}" | sed "/^__require_${section}/,/^__!require_${section}/d"
}

# Remove dangling requires and newlines
remove_unused_requires () {
	file=$1

	# Remove any __require_* or __!require_* leftover lines and
	# merge multiple newlines into a single one (last sed)
	printf "%s\n" "${file}" | sed '/^__.*require_.*/d' | \
		sed 'N;/^\n$/D;P;D;'
}

# Retreive the template file
if test -f $1; then
	template_file=$1
else
	printf "Error: When reading Nanos6 Template Config File\n"
	exit
fi

# Extract a list of all the possible sections from the template file
possible_sections=($(cat $template_file | sed -n 's/^__require_//p'))

# Iterate all the possible sections and if one is not found in parameters, disable it
disabled_sections=()

# We begin with the second parameter, as the first one is the template file
i=2
while [ $i -le $# ]; do
	candidate=${!i}
	found=0
	for section in $@; do
		if [ $candidate == $section ]; then
			found=1
			break
		fi
	done

	if [ !$found ]; then
		disabled_sections+=($section)
	fi

	# Check next section
	i=$((i+1))
done

# For each disabled section, call remove_section
file_contents=$(cat ${template_file})
for section in ${disabled_sections[@]}; do
	file_contents=$(remove_section "${file_contents}" "${section}")
done

# Clean up file
file_contents=$(remove_unused_requires "${file_contents}")

# Return the processed file
printf "%s\n" "$file_contents"
