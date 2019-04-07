#include <mutex>
#include <vector>

#include <nanos6/polling.h>

#include "DataTransferCompletion.hpp"
#include "lowlevel/PaddedSpinLock.hpp"

#include <ClusterManager.hpp>
#include <DataTransfer.hpp>
#include <InstrumentLogMessage.hpp>

// Include these to avoid annoying compiler warnings
#include <InstrumentThreadInstrumentationContextImplementation.hpp>
#include "src/instrument/support/InstrumentThreadLocalDataSupportImplementation.hpp"

namespace ClusterPollingServices {
	namespace {
		struct pending_transfers {
			std::vector<DataTransfer *> _transfers;
			PaddedSpinLock<64> _lock;
		};
		
		struct pending_transfers _pending;
		
		int check_data_transfer(void *service_data)
		{
			struct pending_transfers *pending =
				(struct pending_transfers *)service_data;
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
	}
	
	void addPendingDataTransfer(DataTransfer *dt)
	{
		std::lock_guard<PaddedSpinLock<64>> guard(_pending._lock);
		_pending._transfers.push_back(dt);
	}
	
	void registerDataTransferCompletion()
	{
		nanos6_register_polling_service(
			"cluster data transfer completion",
			check_data_transfer,
			(void *)&_pending
		);
	}
	
	void unregisterDataTransferCompletion()
	{
		nanos6_unregister_polling_service(
			"cluster data transfer completion",
			check_data_transfer,
			(void *)&_pending
		);
#ifndef NDEBUG
		std::lock_guard<PaddedSpinLock<64>> guard(_pending._lock);
		assert(_pending._transfers.empty());
#endif
	}
};
