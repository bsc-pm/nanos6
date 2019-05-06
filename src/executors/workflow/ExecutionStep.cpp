#include "ExecutionStep.hpp"

#include <DataAccess.hpp>

namespace ExecutionWorkflow {
	
	DataLinkStep::DataLinkStep(DataAccess *access) :
		Step(),
		/* We count twice the bytes of the region, because we
		 * need to link both for Read and Write satisfiability */
		_bytes_to_link(2 * access->getAccessRegion().getSize())
	{
	}
	
	DataReleaseStep::DataReleaseStep(DataAccess *access) :
		Step(),
		_type(access->getType()),
		_weak(access->isWeak()),
		_bytes_to_release(access->getAccessRegion().getSize())
	{	
	}
}
