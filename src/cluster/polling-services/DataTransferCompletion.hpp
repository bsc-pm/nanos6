#ifndef DATA_TRANSFER_COMPLETION_HPP
#define DATA_TRANSFER_COMPLETION_HPP

class DataTransfer;

namespace ClusterPollingServices {

	//! \brief Add a pending DataTransfer to the queue
	//!
	//! Adds a DataTransfer to the pending data transfers queue
	//! in order to be polled periodically by the polling
	//! service for completion
	void addPendingDataTransfer(DataTransfer *dt);
	
	//! \brief Initialize the polling service
	void registerDataTransferCompletion();
	
	//! \brief Shutdown the polling service
	void unregisterDataTransferCompletion();
}

#endif /* DATA_TRANSFER_COMPLETION_HPP */
