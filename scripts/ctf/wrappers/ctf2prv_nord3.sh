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

if [ $# -eq 1 ]; then
		comm="ctf2prv"
		args=$@
elif [ $# -eq 2 ]; then
		comm=$1
		args=${@:2}
else
	>&2 echo "usage: $0 <command> <arguments>"
	>&2 echo "       $0 <arguments> # defaults to ctf2prv"
	exit 1
fi

commpath=$(type -P "$comm")
notfound=$?

if [[ "$notfound" == "1" ]]; then
	>&2 echo "The ctf2prv converter is not in the system path. ctf to prv conversion was not possible."
	exit 1
fi

1>&3      echo "Loading Nord3 modules for ctf2prv"
1>&3 2>&3 module purge
1>&3 2>&3 module load gcc/7.2.0 mkl python/3.7.4 swig/3.0.12 babeltrace2/2.0.3
1>&3      echo "Loading modules done"

$commpath $args
