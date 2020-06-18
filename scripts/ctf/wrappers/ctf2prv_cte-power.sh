#!/bin/bash
#
#	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
#
#	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
#

CTF2PRV=`which ctf2prv`

module load swig python/3.6.5 babeltrace2

$CTF2PRV $@
