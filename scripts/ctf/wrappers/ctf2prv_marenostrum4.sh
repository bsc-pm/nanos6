#!/bin/bash
#
#	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
#
#	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
#

set -e

module purge
module load mkl gcc/7.2.0 python/3.7.4 swig/3.0.12 babeltrace2/2.0.3-py3.7.4

ctf2prv $@
