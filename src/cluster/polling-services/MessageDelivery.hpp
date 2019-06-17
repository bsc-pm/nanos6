/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef MESSAGE_DELIVERY_HPP
#define MESSAGE_DELIVERY_HPP

class Message;

namespace ClusterPollingServices {
		
	//! \brief Add a pending message to the queue
	//!
	//! Adds a Message that has been sent in a non-blocking
	//! way to the polling service's queue, in order to be
	//! checked later for completion.
	//!
	//! \param[in] msg is a pending Message
	void addPendingMessage(Message *msg);
	
	//! Initialize the polling service
	void registerMessageDelivery();
	
	//! Shutdown the polling service
	void unregisterMessageDelivery();
}

#endif /* MESSAGE_DELIVERY_HPP */
