#!/bin/bash
#
#	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
#
#	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
#

if [[ "$CTF2PRV_VERBOSE" == "1" ]]; then
	exec 3>&2
else
	exec 3>/dev/null
fi

CTF2PRV=$(type -P "ctf2prv")
NOTFOUND=$?

if [[ "$NOTFOUND" == "1" ]]; then
	>&2 echo "The ctf2prv converter is not in the system path. ctf to prv conversion was not possible.";
	exit 1;
fi

1>&3      echo "Loading CTE-POWER9 modules for ctf2prv"
1>&3 2>&3 module purge
1>&3 2>&3 module load gcc/6.4.0 swig python/3.6.5 babeltrace2
1>&3      echo "Loading modules done"

$CTF2PRV $@
