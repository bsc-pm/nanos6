#include <ExecutionStep.hpp>
#include <DataAccess.hpp>


namespace ExecutionWorkflow {
	
	DataLinkStep::DataLinkStep(DataAccess const *access) :
		Step(),
		_access(access),
		_total_bytes(access->getAccessRegion().getSize()),
		_linked_bytes(0)
	{
	}
	
	DataReleaseStep::DataReleaseStep(DataAccess const *access) :
		Step(),
		_access(access),
		_total_bytes(access->getAccessRegion().getSize()),
		_released_bytes(0)
	{	
	}
}
