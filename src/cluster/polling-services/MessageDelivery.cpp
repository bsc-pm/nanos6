#include <vector>
#include <algorithm>

#include <api/nanos6.h>

#include "MessageDelivery.hpp"
#include "lowlevel/PaddedSpinLock.hpp"

#include <ClusterManager.hpp>
#include <Message.hpp>

namespace ClusterPollingServices {
	
	namespace {
		struct pending_messages {
			std::vector<Message *> _messages;
			PaddedSpinLock<64> _lock;
		};
		
		struct pending_messages _outgoing;
		
		int check_message_delivery(void *service_data)
		{
			struct pending_messages *pending =
				(struct pending_messages *)service_data;
			std::vector<Message *> &messages = pending->_messages;
			
			std::lock_guard<PaddedSpinLock<64>> guard(pending->_lock);
			if (messages.size() > 0) {
				ClusterManager::testMessageCompletion(messages);
				
				messages.erase(
					std::remove_if(
						messages.begin(), messages.end(),
						[](Message *msg) {
							bool delivered = msg->isDelivered();
							if (delivered) {
								delete msg;
							}
							
							return delivered;
						}
					),
					std::end(messages)
				);
			}
			
			//! We will only unregister this service from the
			//! ClusterManager at shutdown
			return 0;
		}
	};
	
	void addPendingMessage(Message *msg)
	{
		std::lock_guard<PaddedSpinLock<64>> guard(_outgoing._lock);
		_outgoing._messages.push_back(msg);
	}
	
	void registerMessageDelivery()
	{
		nanos6_register_polling_service(
			"cluster message delivery",
			check_message_delivery,
			(void *)&_outgoing
		);
	}
	
	void unregisterMessageDelivery()
	{
		nanos6_unregister_polling_service(
			"cluster message delivery",
			check_message_delivery,
			(void *)&_outgoing
		);
#ifndef NDEBUG
		std::lock_guard<PaddedSpinLock<64>> guard(_outgoing._lock);
		assert(_outgoing._messages.empty());
#endif
	}
};
