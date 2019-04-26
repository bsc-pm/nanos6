#include "MessageDfree.hpp"

#include <ClusterManager.hpp>
#include <DataAccessRegion.hpp>
#include <DistributionPolicy.hpp>

MessageDfree::MessageDfree(const ClusterNode *from) :
	Message("MessageDfree", DFREE, sizeof(DfreeMessageContent), from)
{
	_content = reinterpret_cast<DfreeMessageContent *>(_deliverable->payload);
}

bool MessageDfree::handleMessage()
{
	DataAccessRegion region(_content->_address, _content->_size);
	
	//! TODO: We need to fix the way we allocate distributed memory so that
	//! we do allocate it from the MemoryAllocator instead of the
	//! VirtualMemoryManagement layer, which is what we do now. The
	//! VirtualMemoryManagement layer does not allow (at the moment)
	//! deallocation of memory, so for now we do not free distributed
	//! memory
	
	//! Unregister the region from the home node map
	ClusterDirectory::unregisterAllocation(region);
	
	ClusterManager::synchronizeAll();
	
	return true;
}

//! Register the Message type to the Object factory
static Message *createDfreeMessage(Message::Deliverable *dlv)
{
	return new MessageDfree(dlv);
}

static const bool __attribute__((unused))_registered_dfree =
	REGISTER_MSG_CLASS(DFREE, createDfreeMessage);
