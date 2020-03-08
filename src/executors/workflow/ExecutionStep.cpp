/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#include "ExecutionStep.hpp"

#include <DataAccess.hpp>

namespace ExecutionWorkflow {
	
	DataLinkStep::DataLinkStep(DataAccess *access) :
		Step(),
		/* We count twice the bytes of the region, because we
		 * need to link both for Read and Write satisfiability */
		_bytesToLink(2 * access->getAccessRegion().getSize())
	{
	}
	
	DataReleaseStep::DataReleaseStep(DataAccess *access) :
		Step(),
		_type(access->getType()),
		_weak(access->isWeak()),
		_bytesToRelease(access->getAccessRegion().getSize())
	{
	}
}
