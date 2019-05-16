#include "ExecutionStep.hpp"

#include <DataAccess.hpp>

namespace ExecutionWorkflow {
	
	DataLinkStep::DataLinkStep(DataAccess const *access) :
		Step(),
		_access(access),
		/* We count twice the bytes of the region, because we
		 * need to link both for Read and Write satisfiability */
		_bytes_to_link(2 * access->getAccessRegion().getSize())
	{
	}
	
	DataReleaseStep::DataReleaseStep(DataAccess const *access) :
		Step(),
		_access(access),
		_bytes_to_release(access->getAccessRegion().getSize())
	{	
	}
	
	void DataReleaseStep::start()
	{
		releaseSuccessors();
		delete this;
	}
}
