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
module load mkl gcc/7.2.0 python/3.7.4 swig/3.0.12 babeltrace2/2.0.3-py3.7.4  2>/dev/null

$CTF2PRV $@
