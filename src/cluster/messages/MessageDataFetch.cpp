/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#include "MessageDataFetch.hpp"

#include <ClusterManager.hpp>
#include <DataTransferCompletion.hpp>

MessageDataFetch::MessageDataFetch(const ClusterNode *from,
		DataAccessRegion const &remoteRegion)
	: Message("MessageDataFetch", DATA_FETCH,
		sizeof(DataFetchMessageContent), from)
{
	_content = reinterpret_cast<DataFetchMessageContent *>(_deliverable->payload);
	_content->_remoteRegion = remoteRegion;
}

bool MessageDataFetch::handleMessage()
{
	ClusterMemoryNode *memoryPlace =
		ClusterManager::getMemoryNode(getSenderId());
	
	DataTransfer *dt =
		ClusterManager::sendDataRaw(_content->_remoteRegion, memoryPlace,
				getId());
	
	ClusterPollingServices::addPendingDataTransfer(dt);
	
	return true;
}

static Message *createDataFetchMessage(Message::Deliverable *dlv)
{
	return new MessageDataFetch(dlv);
}

static const bool __attribute__((unused))_registered_dfetch =
	REGISTER_MSG_CLASS(DATA_FETCH, createDataFetchMessage);
