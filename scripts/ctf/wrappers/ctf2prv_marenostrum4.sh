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

1>&3      echo "Loading Marenostrum modules for ctf2prv"
1>&3 2>&3 module purge
1>&3 2>&3 module load mkl gcc/7.2.0 python/3.7.4 swig/3.0.12 babeltrace2/2.0.3-py3.7.4
1>&3      echo "Loading modules done"

$CTF2PRV $@
