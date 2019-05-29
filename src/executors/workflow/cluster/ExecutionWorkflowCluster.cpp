#include "ExecutionWorkflowCluster.hpp"
#include "tasks/Task.hpp"

#include <ClusterManager.hpp>
#include <ClusterPollingServices.hpp>
#include <DataAccess.hpp>
#include <DataTransfer.hpp>
#include <Directory.hpp>
#include <InstrumentLogMessage.hpp>
#include <TaskOffloading.hpp>

namespace ExecutionWorkflow {
 	
	void ClusterDataLinkStep::linkRegion(
		DataAccessRegion const &region,
		MemoryPlace const *location,
		bool read,
		bool write
	) {
		assert(_targetMemoryPlace != nullptr);
		TaskOffloading::SatisfiabilityInfo satInfo(region,
			location->getIndex(), read, write);
		
		TaskOffloading::ClusterTaskContext *clusterTaskContext =
			_task->getClusterContext();
		TaskOffloading::sendSatisfiability(_task,
			clusterTaskContext->getRemoteNode(), satInfo);
		size_t linkedBytes = region.getSize();
		
		//! We need to account for linking both read and write
		//! satisfiability
		if (read && write) {
			linkedBytes *= 2;
		}
		if ((_bytesToLink -= linkedBytes) == 0) {
			delete this;
		}
	}
	
	void ClusterDataLinkStep::start()
	{
		assert(_targetMemoryPlace != nullptr);
		
		if (!_read && !_write) {
			//! Nothing to do here. We can release the execution
			//! step. Location will be linked later on.
			releaseSuccessors();
			return;
		}
		
		assert(_sourceMemoryPlace != nullptr);
		Instrument::logMessage(
			Instrument::ThreadInstrumentationContext::getCurrent(),
			"ClusterDataLinkStep for MessageTaskNew. ",
			"Current location of ", _region,
			" Node:", _sourceMemoryPlace->getIndex()
		);
		
		//! The current node is the source node. We just propagate
		//! the info we 've gathered
		assert(_successors.size() == 1);
		ClusterExecutionStep *execStep =
			(ClusterExecutionStep *)_successors[0];
		
		assert(_read || _write);
		execStep->addDataLink(_sourceMemoryPlace->getIndex(),
			_region, _read, _write);
		
		releaseSuccessors();
		size_t linkedBytes = _region.getSize();
		//! If at the moment of offloading the access is not both
		//! read and write satisfied, then the info will be linked
		//! later on. In this case, we just account for the bytes that
		//! we link now, the Step will be deleted when all the bytes
		//! are linked through linkRegion method invocation
		if (_read && _write) {
			delete this;
		} else {
			_bytesToLink -= linkedBytes;
		}
	}
	
	void ClusterDataReleaseStep::releaseRegion(
		DataAccessRegion const &region, MemoryPlace const *location
	) {
		Instrument::logMessage(
			Instrument::ThreadInstrumentationContext::getCurrent(),
			"releasing remote region:", region);
		TaskOffloading::sendRemoteAccessRelease(_remoteTaskIdentifier,
				_offloader, region, _type, _weak, location);
		
		if ((_bytesToRelease -= region.getSize()) == 0) {
			delete this;
		}
	}
	
	bool ClusterDataReleaseStep::checkDataRelease(DataAccess const *access)
	{
		bool releases = (access->getObjectType() == taskwait_type)
			&& access->getOriginator()->isSpawned()
			&& access->readSatisfied()
			&& access->writeSatisfied();

		Instrument::logMessage(
			Instrument::ThreadInstrumentationContext::getCurrent(),
			"Checking DataRelease access:",
			access->getInstrumentationId(),
			" object_type:", access->getObjectType(),
			" spawned originator:", access->getOriginator()->isSpawned(),
			" read:", access->readSatisfied(),
			" write:", access->writeSatisfied(),
			" releases:", releases);
		
		return releases;
	}
	
	void ClusterDataReleaseStep::start()
	{
		releaseSuccessors();
	}
	
	void ClusterDataCopyStep::start()
	{
		assert(ClusterManager::getCurrentMemoryNode() == _targetMemoryPlace);
		
		//! No data transfer needed, data is already here.
		if (_sourceMemoryPlace == _targetMemoryPlace) {
			releaseSuccessors();
			delete this;
			return;
		}
		
		DataTransfer *dt;
		Instrument::logMessage(
			Instrument::ThreadInstrumentationContext::getCurrent(),
			"ClusterDataCopyStep fetching data from Node:",
			_sourceMemoryPlace->getIndex()
		);
		dt = ClusterManager::fetchData(
			_targetTranslation._hostRegion,
			_sourceMemoryPlace);
		
		dt->setCompletionCallback(
			[&]() {
				this->releaseSuccessors();
				delete this;
			}
		);
		
		ClusterPollingServices::addPendingDataTransfer(dt);
	}
	
	ClusterExecutionStep::ClusterExecutionStep(
		Task *task,
		ComputePlace *computePlace
	) : Step(),
		_satInfo(),
		_remoteNode(reinterpret_cast<ClusterNode *>(computePlace)),
		_task(task)
	{
		assert(computePlace->getType() == nanos6_cluster_device);
		TaskOffloading::ClusterTaskContext *clusterContext =
			new TaskOffloading::ClusterTaskContext((void *)_task,
					_remoteNode);
		_task->setClusterContext(clusterContext);
	}
	
	void ClusterExecutionStep::addDataLink(int source,
		DataAccessRegion const &region, bool read, bool write)
	{
		std::lock_guard<SpinLock> guard(_lock);
		_satInfo.push_back(
			TaskOffloading::SatisfiabilityInfo(region, source,
				read, write)
		);
	}
	
	void ClusterExecutionStep::start()
	{
		_task->setExecutionStep(this);
		TaskOffloading::offloadTask(_task, _satInfo, _remoteNode);
	}
	
	void ClusterNotificationStep::start()
	{
		if (_callback) {
			_callback();
		}
		
		releaseSuccessors();
		delete this;
	}
};
