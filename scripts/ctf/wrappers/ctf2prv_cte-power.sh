#!/bin/bash
#
#	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
#
#	Copyright (C) 2021 Barcelona Supercomputing Center (BSC)
#

if [[ "$CTF2PRV_VERBOSE" == "1" ]]; then
	exec 3>&2
else
	exec 3>/dev/null
fi

if [ $# -eq 0 ]; then
	>&2 echo "$0: bad usage"
	>&2 echo "usage: $0 command [args]"
	exit 1
fi

comm="$1"
shift

commpath=$(type -P "$comm")
notfound=$?

if [[ "$notfound" == "1" ]]; then
	>&2 echo "Cannot find $comm in the system path"
	exit 1
fi

1>&3      echo "Loading CTE-POWER9 modules for ctf2prv"
1>&3 2>&3 module purge
1>&3 2>&3 module load gcc/6.4.0 swig python/3.6.5 babeltrace2
1>&3      echo "Loading modules done"

"$commpath" "$@"
