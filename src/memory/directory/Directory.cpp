/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#include "Directory.hpp"

HomeNodeMap Directory::_homeNodes;
MemoryPlace Directory::_directoryMemoryPlace(-42, nanos6_host_device);
