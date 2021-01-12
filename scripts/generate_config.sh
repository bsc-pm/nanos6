#!/bin/sh -e
#
#	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
#
#	Copyright (C) 2021 Barcelona Supercomputing Center (BSC)
#
# Generate config depending on enabled flags

# Retreive the template file
if [ -z "$1" ]; then
	>&2 echo "Error: When reading Nanos6 Template Config File"
	>&2 echo "Usage: generate_config.sh file [sections_to_remove]"
	exit 1
fi

template="$1"
shift

sectionFilter=$(echo "$@" | sed 's/ /|/g')

# Remove stated sections
sed -E '/^__require_('"$sectionFilter"')$/,/^__!require_.*$/d' "$template" | \
	# Remove leftover "require" lines
	sed '/^__!\?require_.*$/d' | \
	# Merge multiple newlines into one
	sed 'N;/^\n$/D;P;D;'
