#include <mutex>
#include <vector>

#include <nanos6/polling.h>

#include "DataTransferCompletion.hpp"
#include "lowlevel/PaddedSpinLock.hpp"

#include <ClusterManager.hpp>
#include <DataTransfer.hpp>
#include <InstrumentLogMessage.hpp>


namespace ClusterPollingServices {
	struct PendingTransfers {
		std::vector<DataTransfer *> _transfers;
		PaddedSpinLock<64> _lock;
	};
	
	static PendingTransfers _pendingTransfers;
	
	static int checkDataTransfers(void *service_data)
	{
		PendingTransfers *pending =
			(PendingTransfers *)service_data;
		assert(pending != nullptr);
		
		std::vector<DataTransfer *> &transfers = pending->_transfers;
		
		std::lock_guard<PaddedSpinLock<64>> guard(pending->_lock);
		if (transfers.size() == 0) {
			//! We will only unregister this service from the
			//! ClusterManager at shutdown
			return 0;
		}
		
		ClusterManager::testDataTransferCompletion(transfers);
		transfers.erase(
			std::remove_if(
				transfers.begin(), transfers.end(),
				[](DataTransfer *dt) {
					assert(dt != nullptr);
					
					bool completed = dt->isCompleted();
					if (completed) {
						delete dt;
					}
					
					return completed;
				}
			),
			std::end(transfers)
		);
		
		//! We will only unregister this service from the
		//! ClusterManager at shutdown
		return 0;
	}
	
	void addPendingDataTransfer(DataTransfer *dt)
	{
		std::lock_guard<PaddedSpinLock<64>> guard(_pendingTransfers._lock);
		_pendingTransfers._transfers.push_back(dt);
	}
	
	void registerDataTransferCompletion()
	{
		nanos6_register_polling_service(
			"cluster data transfer completion",
			checkDataTransfers,
			(void *)&_pendingTransfers
		);
	}
	
	void unregisterDataTransferCompletion()
	{
		nanos6_unregister_polling_service(
			"cluster data transfer completion",
			checkDataTransfers,
			(void *)&_pendingTransfers
		);
		
#ifndef NDEBUG
		std::lock_guard<PaddedSpinLock<64>> guard(_pendingTransfers._lock);
		assert(_pendingTransfers._transfers.empty());
#endif
	}
}
