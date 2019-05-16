/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#include "MessageDmalloc.hpp"

#include <ClusterManager.hpp>
#include <ClusterMemoryManagement.hpp>
#include <DistributionPolicy.hpp>
#include <VirtualMemoryManagement.hpp>

MessageDmalloc::MessageDmalloc(const ClusterNode *from, size_t numDimensions)
	: Message("MessageDmalloc", DMALLOC,
		sizeof(DmallocMessageContent) + numDimensions * sizeof(size_t),
		from)
{
	_content = reinterpret_cast<DmallocMessageContent *>(_deliverable->payload);
}

bool MessageDmalloc::handleMessage()
{
	void *dptr = nullptr;
	size_t size = getAllocationSize();
	nanos6_data_distribution_t policy = getDistributionPolicy();
	size_t nrDim = getDimensionsSize();
	size_t *dimensions = getDimensions();
	
	if (ClusterManager::isMasterNode()) {
		/* The master node performs the allocation and communicates
		 * the allocated address to all other nodes */
		dptr = VirtualMemoryManagement::allocDistrib(size);
		
		DataAccessRegion address(&dptr, sizeof(void *));
		
		ClusterNode *current = ClusterManager::getCurrentClusterNode();
		assert(current != nullptr);
		
		std::vector<ClusterNode *> const &world =
			ClusterManager::getClusterNodes();
		
		/* Send the allocated address to everyone else */
		for (ClusterNode *node : world) {
			if (node == current) {
				continue;
			}
			
			ClusterMemoryNode *memoryNode = node->getMemoryNode();
			ClusterManager::sendDataRaw(address, memoryNode, getId(),
						true);
		}
	} else {
		/* This is a slave node. We will receive the allocated address
		 * from the master node */
		DataAccessRegion address(&dptr, sizeof(void *));
		
		ClusterNode *masterNode = ClusterManager::getMasterNode();
		ClusterMemoryNode *masterMemoryNode =
			masterNode->getMemoryNode();
		
		ClusterManager::fetchDataRaw(address, masterMemoryNode, getId(),
						true);
	}
	
	/* Register the newly allocated region with the Directory
	 * of home nodes */
	DataAccessRegion allocatedRegion(dptr, size);
	ClusterDirectory::registerAllocation(allocatedRegion, policy, nrDim,
				dimensions);
	
	ClusterManager::synchronizeAll();
	
	return true;
}

//! Register the Message type to the Object factory
static Message *createDmallocMessage(Message::Deliverable *dlv)
{
	return new MessageDmalloc(dlv);
}

static const bool __attribute__((unused))_registered_dmalloc =
	REGISTER_MSG_CLASS(DMALLOC, createDmallocMessage);
