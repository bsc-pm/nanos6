#include <cassert>

#include "ExecutionWorkflow.hpp"
#include "executors/threads/TaskFinalization.hpp"
#include "lowlevel/FatalErrorHandler.hpp"
#include "tasks/TaskImplementation.hpp"

#include <DataAccess.hpp>
#include <DataAccessRegistration.hpp>
#include <ExecutionWorkflowHost.hpp>
#include <ExecutionWorkflowCluster.hpp>
#include <HardwareCounters.hpp>
#include <Monitoring.hpp>


namespace ExecutionWorkflow {
	
	transfers_map_t _transfersMap {
			/*  host      cuda     opencl    cluster   */
	/* host */	{ nullCopy, nullCopy, nullCopy, clusterCopy },
	/* cuda */	{ nullCopy, nullCopy, nullCopy, nullCopy },
	/* opencl */	{ nullCopy, nullCopy, nullCopy, nullCopy },
	/* cluster */	{ clusterCopy, nullCopy, nullCopy, clusterCopy }
	};
	
	Step *WorkflowBase::createAllocationAndPinningStep(
		RegionTranslation &regionTranslation,
		MemoryPlace const *memoryPlace
	) {
		switch (memoryPlace->getType()) {
			case nanos6_host_device:
				return new HostAllocationAndPinningStep(
						regionTranslation, memoryPlace);
			case nanos6_cluster_device:
				return new ClusterAllocationAndPinningStep(
						regionTranslation, memoryPlace);
			case nanos6_cuda_device:
			case nanos6_opencl_device:
			default:
				FatalErrorHandler::failIf(
					true,
					"Execution workflow does not support this "
					"device yet"
				);
				break;
		}
		
		//! Silencing annoying compiler warning
		return nullptr;
	}

	Step *WorkflowBase::createDataCopyStep(
		MemoryPlace const *sourceMemoryPlace,
		MemoryPlace const *targetMemoryPlace,
		RegionTranslation const &targetTranslation,
		DataAccess *access
	) {
		/* At the moment we do not support data copies for accesses
		 * of the following types. This essentially mean that devices,
		 * e.g. Cluster, CUDA, do not support these accesses. */
		if (access->getType() == REDUCTION_ACCESS_TYPE ||
			access->getType() == COMMUTATIVE_ACCESS_TYPE ||
			access->getType() == CONCURRENT_ACCESS_TYPE
		) {
			return new Step();
		}
		
		assert(targetMemoryPlace != nullptr);
		//assert(sourceMemoryPlace != nullptr);
		
		nanos6_device_t sourceType =
			(sourceMemoryPlace == nullptr)
				? nanos6_host_device
				: sourceMemoryPlace->getType();
		nanos6_device_t targetType = targetMemoryPlace->getType();
		
		return _transfersMap[sourceType][targetType](
				sourceMemoryPlace,
				targetMemoryPlace,
				targetTranslation,
				access);
	}
	
	Step *WorkflowBase::createExecutionStep(Task *task, ComputePlace *computePlace)
	{
		switch(computePlace->getType()) {
			case nanos6_host_device:
				return new HostExecutionStep(task, computePlace);
			case nanos6_cluster_device:
				return new ClusterExecutionStep(task, computePlace);
			case nanos6_cuda_device:
			case nanos6_opencl_device:
			default:
				FatalErrorHandler::failIf(
					true,
					"Execution workflow does not support "
					"this device yet"
				);
				break;
		}
		
		//! Silencing annoying compiler warning
		return nullptr;
	}
	
	Step *WorkflowBase::createNotificationStep(
		std::function<void ()> const &callback,
		ComputePlace *computePlace
	) {
		nanos6_device_t type =
			(computePlace == nullptr) ?
				nanos6_host_device : computePlace->getType();
		
		switch (type) {
			case nanos6_host_device:
				return new HostNotificationStep(callback);
			case nanos6_cluster_device:
				return new ClusterNotificationStep(callback);
			case nanos6_cuda_device:
			case nanos6_opencl_device:
			default:
				FatalErrorHandler::failIf(
					true,
					"Execution workflow does not support "
					"this device yet"
				);
				break;
		}
		
		//! Silencing annoying compiler warning
		return nullptr;
	}
	
	Step *WorkflowBase::createDataReleaseStep(
		Task const *task,
		DataAccess *access
	) {
		if (task->isRemote()) {
			return new ClusterDataReleaseStep(
					task->getClusterContext(), access);
		}
		
		return new DataReleaseStep(access);
	}
	
	Step *WorkflowBase::createUnpinningStep(MemoryPlace const *targetMemoryPlace,
			RegionTranslation const &targetTranslation)
	{
		switch (targetMemoryPlace->getType()) {
			case nanos6_host_device:
				return new HostUnpinningStep(targetMemoryPlace, targetTranslation);
			case nanos6_cluster_device:
				return new ClusterUnpinningStep(targetMemoryPlace, targetTranslation);
			case nanos6_cuda_device:
			case nanos6_opencl_device:
			default:
				FatalErrorHandler::failIf(
					true,
					"Execution workflow does not support "
					"this device yet"
				);
				break;
		}
		
		//! Silencing annoying compiler warning
		return nullptr;
	}
	
	void WorkflowBase::start()
	{
		for (Step *step : _rootSteps) {
			step->start();
		}
	}
	
	void executeTask(
		Task *task,
		ComputePlace *targetComputePlace,
		MemoryPlace *targetMemoryPlace
	) {
		/* The workflow has already been created for this Task.
		 * At this point the Task has been assigned to a WorkerThread
		 * because all its pending DataCopy steps have been completed
		 * and it's ready to actually run */
		if (task->getWorkflow() != nullptr) {
			ExecutionWorkflow::Step *executionStep =
				task->getExecutionStep();
			
			assert(executionStep != nullptr);
			executionStep->start();
			
			return;
		}
		
		//! This is the target MemoryPlace that we will use later on,
		//! once the Task has completed, to update the location of its
		//! DataAccess objects. This can be overriden, if we
		//! release/unregister the accesses passing a different
		//! MemoryPlace.
		task->setMemoryPlace(targetMemoryPlace);
		
		//int numSymbols = task->getSymbolNum();
		Workflow<TaskExecutionWorkflowData> *workflow =
			createWorkflow<TaskExecutionWorkflowData>(0 /* numSymbols */);
		
		Step *executionStep =
			workflow->createExecutionStep(task, targetComputePlace);
		
		Step *notificationStep = workflow->createNotificationStep(
			[=]() {
				WorkerThread *currThread = WorkerThread::getCurrentWorkerThread();
				
				CPU *cpu = nullptr;
				if (currThread != nullptr) {
					cpu = currThread->getComputePlace();
				}
				
				CPUDependencyData localDependencyData;
				CPUDependencyData &hpDependencyData = (cpu != nullptr) ?
					cpu->getDependencyData() : localDependencyData;
				
				if (task->markAsFinished(cpu/* cpu */)) {
					DataAccessRegistration::unregisterTaskDataAccesses(
						task,
						cpu, /*cpu, */
						hpDependencyData,
						targetMemoryPlace
					);
					
					Monitoring::taskFinished(task);
					HardwareCounters::taskFinished(task);
					
					task->setComputePlace(nullptr);
					
					if (task->markAsReleased()) {
						TaskFinalization::disposeOrUnblockTask(task, cpu);
					}
				}
				
				delete workflow;
			},
			targetComputePlace
		);
		
		TaskDataAccesses &accessStructures = task->getDataAccesses();
		
		{
			std::lock_guard<TaskDataAccesses::spinlock_t>
				guard(accessStructures._lock);
			
			/* TODO: Once we have correct management for the Task symbols here
			 * we should create the corresponding allocation steps. */
			
			accessStructures._accesses.processAll(
				[&](TaskDataAccesses::accesses_t::iterator position) -> bool {
					DataAccess *dataAccess = &(*position);
					assert(dataAccess != nullptr);
					DataAccessRegion region = dataAccess->getAccessRegion();
					
					MemoryPlace const *currLocation = dataAccess->getLocation();
					/* TODO: This will be provided by the corresponding
					 * AllocationAndPinning step, once we fix this functionality.
					 * At the moment (and since we support only cluster and SMP
					 * we can use a dummy RegionTranslation */
					RegionTranslation translation(region, region.getStartAddress());
					Step *step = workflow->createDataCopyStep(
							currLocation,
							targetMemoryPlace,
							translation,
							dataAccess);
					
					workflow->enforceOrder(step, executionStep);
					workflow->addRootStep(step);
					
					step = workflow->createDataReleaseStep(task,
							dataAccess);
					workflow->enforceOrder(executionStep, step);
					workflow->enforceOrder(step, notificationStep);
					
					return true;
				}
			);
		}
		
		if (executionStep->ready()) {
			workflow->enforceOrder(executionStep, notificationStep);
			workflow->addRootStep(executionStep);
		}
		
		task->setWorkflow(workflow);
		task->setComputePlace(targetComputePlace);
		
		//! Starting the workflow will either execute the task to
		//! completion (if there are not pending transfers for the
		//! task), or it will setup all the Execution Step will
		//! execute when ready.
		workflow->start();
	}
	
	void setupTaskwaitWorkflow(
		Task *task,
		DataAccess *taskwaitFragment
	) {
		ComputePlace *computePlace = nullptr;
		WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
		if (currentThread != nullptr) {
			computePlace = currentThread->getComputePlace();
		}
		
		ExecutionWorkflow::Workflow<RegionTranslation> *workflow =
			createWorkflow<RegionTranslation>();
		
		DataAccessRegion region = taskwaitFragment->getAccessRegion();
		
		//! This for the time works, but probably for devices with address
		//! translation we might need to revise it.
		RegionTranslation translation(region, region.getStartAddress());
		
		Step *notificationStep =
			workflow->createNotificationStep(
				[=]() {
					/* We cannot re-use the 'computePlace', we need to
					 * retrieve the current Thread and associated
					 * ComputePlace */
					ComputePlace *releasingComputePlace = nullptr;
					WorkerThread *releasingThread = WorkerThread::getCurrentWorkerThread();
					if (releasingThread != nullptr) {
						releasingComputePlace = releasingThread->getComputePlace();
					}
					
					/* Here, we are always using a local CPUDependencyData
					 * object, to avoid the issue where we end-up calling
					 * this while the thread is already in the dependency
					 * system, using the CPUDependencyData of its
					 * ComputePlace. This is a *TEMPORARY* solution, until
					 * we fix how we handle taskwaits in a more clean 
					 * way. */
					CPUDependencyData localDependencyData;
					
					DataAccessRegistration::releaseTaskwaitFragment(
						task,
						region,
						releasingComputePlace,
						localDependencyData
					);
					
					delete workflow;
				},
				computePlace
			);
		
		MemoryPlace const *currLocation =
			taskwaitFragment->getLocation();
		MemoryPlace const *targetLocation =
			taskwaitFragment->getOutputLocation();
		
		//! No need to perform any copy for this taskwait fragment
		if (targetLocation == nullptr) {
			workflow->addRootStep(notificationStep);
			workflow->start();
			return;
		}
		
		Step *copyStep =
			workflow->createDataCopyStep(
				currLocation,
				targetLocation,
				translation,
				taskwaitFragment
			);
		
		workflow->addRootStep(copyStep);
		workflow->enforceOrder(copyStep, notificationStep);
		workflow->start();
	}
};
