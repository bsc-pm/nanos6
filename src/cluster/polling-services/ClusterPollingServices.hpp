/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef CLUSTER_POLLING_SERVICES_HPP
#define CLUSTER_POLLING_SERVICES_HPP

#include "DataTransferCompletion.hpp"
#include "MessageDelivery.hpp"
#include "MessageHandler.hpp"

namespace ClusterPollingServices {
	
	//! \brief Initialize the Cluster polling services
	//!
	//! This method will be called during ClusterManager
	//! initialization.
	//! New type of polling services need to expose an
	//! initialization interface that will be called from here
	inline void initialize()
	{
		registerMessageHandler();
		registerMessageDelivery();
		registerDataTransferCompletion();
	}
	
	//! \brief Shutdown the Cluster polling services
	//!
	//! This method will be called during ClusterManager
	//! shutdown.
	//! New type of polling services need to expose a
	//! shutdown interface that will be called from here.
	inline void shutdown()
	{
		unregisterDataTransferCompletion();
		unregisterMessageDelivery();
		unregisterMessageHandler();
	}
}


#endif /* CLUSTER_POLLING_SERVICES_HPP */
