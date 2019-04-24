#include <vector>
#include <algorithm>

#include <api/nanos6.h>

#include "MessageDelivery.hpp"
#include "lowlevel/PaddedSpinLock.hpp"

#include <ClusterManager.hpp>
#include <Message.hpp>

namespace ClusterPollingServices {
	struct PendingMessages {
		std::vector<Message *> _messages;
		PaddedSpinLock<64> _lock;
	};
	
	static PendingMessages _outgoingMessages;
	
	static int checkMessageDelivery(void *service_data)
	{
		PendingMessages *pending = (PendingMessages *)service_data;
		assert(pending != nullptr);
		
		std::vector<Message *> &messages = pending->_messages;
		
		std::lock_guard<PaddedSpinLock<64>> guard(pending->_lock);
		if (messages.size() == 0) {
			//! We will only unregister this service from the
			//! ClusterManager at shutdown
			return 0;
		}
		
		ClusterManager::testMessageCompletion(messages);
		
		messages.erase(
			std::remove_if(
				messages.begin(), messages.end(),
				[](Message *msg) {
					assert(msg != nullptr);
					
					bool delivered = msg->isDelivered();
					if (delivered) {
						delete msg;
					}
					
					return delivered;
				}
			),
			std::end(messages)
		);
	
		//! We will only unregister this service from the
		//! ClusterManager at shutdown
		return 0;
	}
	
	void addPendingMessage(Message *msg)
	{
		std::lock_guard<PaddedSpinLock<64>> guard(_outgoingMessages._lock);
		_outgoingMessages._messages.push_back(msg);
	}
	
	void registerMessageDelivery()
	{
		nanos6_register_polling_service(
			"cluster message delivery",
			checkMessageDelivery,
			(void *)&_outgoingMessages
		);
	}
	
	void unregisterMessageDelivery()
	{
		nanos6_unregister_polling_service(
			"cluster message delivery",
			checkMessageDelivery,
			(void *)&_outgoingMessages
		);
		
#ifndef NDEBUG
		std::lock_guard<PaddedSpinLock<64>> guard(_outgoingMessages._lock);
		assert(_outgoingMessages._messages.empty());
#endif
	}
}
