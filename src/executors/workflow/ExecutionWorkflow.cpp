#include <ExecutionWorkflow.hpp>
#include <ExecutionWorkflowHost.hpp>
#include "lowlevel/FatalErrorHandler.hpp"
#include <DataAccessRegistration.hpp>
#include <DataAccess.hpp>
#include "executors/threads/TaskFinalization.hpp"

#include <cassert>

// Include these to avoid annoying compiler warnings
#include <InstrumentThreadInstrumentationContextImplementation.hpp>
#include "src/instrument/support/InstrumentThreadLocalDataSupportImplementation.hpp"

namespace ExecutionWorkflow {
	
	transfers_map_t _transfersMap {
			/*  host      cuda     opencl    cluster   */
	/* host */	{ nullCopy, nullCopy, nullCopy, nullCopy },
	/* cuda */	{ nullCopy, nullCopy, nullCopy, nullCopy },
	/* opencl */	{ nullCopy, nullCopy, nullCopy, nullCopy },
	/* cluster */	{ nullCopy, nullCopy, nullCopy, nullCopy }
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
		
		/* the source MemoryPlace might be NULL at this point,
		 * in the case of weak accesses */
		nanos6_device_t sourceType = (sourceMemoryPlace != nullptr) ?
			sourceMemoryPlace->getType() : nanos6_host_device;
		
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
		__attribute__((unused))Task const *task,
		DataAccess *access
	) {
		return new DataReleaseStep(access);
	}
	
	Step *WorkflowBase::createUnpinningStep(MemoryPlace const *targetMemoryPlace,
			RegionTranslation const &targetTranslation)
	{
		switch (targetMemoryPlace->getType()) {
			case nanos6_host_device:
				return new HostUnpinningStep(targetMemoryPlace, targetTranslation);
			case nanos6_cluster_device:
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
		
		//int numSymbols = task->getSymbolNum();
		Workflow<TaskExecutionWorkflowData> *workflow =
			createWorkflow<TaskExecutionWorkflowData>(0 /* numSymbols */);
		
		Step *executionStep =
			workflow->createExecutionStep(task, targetComputePlace);
		
		Step *notificationStep = workflow->createNotificationStep(
			[=]() {
				WorkerThread *currThread = WorkerThread::getCurrentWorkerThread();
				CPUDependencyData hpDependencyData;
				
				/* At the moment unregistering accesses should happen
				 * using a NULL cpu because if this is called from
				 * within a polling service. However calling the
				 * disposeOrUnblockTask call *NEEDS* to have a valid CPU */
				CPU *cpu = nullptr;
				if (currThread != nullptr) {
					cpu = currThread->getComputePlace();
				}
				if (task->markAsFinished(cpu/* cpu */)) {
					DataAccessRegistration::unregisterTaskDataAccesses(
						task,
						cpu, /*cpu, */
						hpDependencyData,
						targetMemoryPlace
					);
					
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
					
					MemoryPlace *currLocation = dataAccess->getLocation();
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
					CPUDependencyData hpDependencyData;
					DataAccessRegistration::releaseTaskwaitFragment(
						task,
						region,
						computePlace,
						hpDependencyData
					);
					
					delete workflow;
				},
				computePlace
			);
		
		MemoryPlace *currLocation = taskwaitFragment->getLocation();
		MemoryPlace *targetLocation = taskwaitFragment->getOutputLocation();
		
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
