/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef MESSAGE_HANDLER_HPP
#define MESSAGE_HANDLER_HPP

namespace ClusterPollingServices {

	//! \brief Initialize the polling service
	void registerMessageHandler();
	
	//! \brief Shutdown the polling service
	void unregisterMessageHandler();
}

#endif /* MESSAGE_HANDLER_HPP */
