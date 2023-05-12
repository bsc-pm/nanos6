#!/bin/sh -e
#	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
#
#	Copyright (C) 2023 Barcelona Supercomputing Center (BSC)

if ! command -v autoreconf; then
	echo "error: autoreconf not found"
	exit 1
fi

autoreconf -fiv
