#!/bin/bash
#
#	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
#
#	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
#

set -e

CTF2PRV=$(type -P "ctf2prv")
FOUND=$?

if [[ ! $FOUND ]]; then
	>&2 echo "The ctf2prv converter is not in the system path. ctf to prv conversion was not possible.";
	exit 1;
fi

module purge  2>/dev/null
module load gcc/6.4.0 swig python/3.6.5 babeltrace2  2>/dev/null

$CTF2PRV $@
