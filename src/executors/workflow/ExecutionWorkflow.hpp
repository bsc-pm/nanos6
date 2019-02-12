#ifndef __EXECUTION_WORKFLOW_HPP__
#define __EXECUTION_WORKFLOW_HPP__

#include <functional>
#include <mutex>
#include <vector>

#include "ExecutionStep.hpp"

class ComputePlace;
class DataAccess;
class MemoryPlace;
class Task;

namespace ExecutionWorkflow {
	
	//! A function that sets up a data transfer between two MemoryPlace
	typedef std::function<Step *(MemoryPlace const *, MemoryPlace const *,
		RegionTranslation const &, DataAccess *)> data_transfer_function_t;
	
	//! A map that stores the functions that perform data transfers between
	//! two MemoryPlaces, depending on their type (nanos6_device_t).
	typedef std::vector<std::vector<data_transfer_function_t> > transfers_map_t;
	
	inline Step *nullCopy(
		__attribute__((unused))MemoryPlace const *source,
		__attribute__((unused))MemoryPlace const *target,
		__attribute__((unused))RegionTranslation const &translation,
		__attribute__((unused))DataAccess *access
	) {
		return new Step();
	}
	
	extern transfers_map_t _transfersMap;
	
	class WorkflowBase {
		//! Root steps of the workflow
		std::vector<Step *> _rootSteps;
		
	public:
		//! \brief Creates an AllocationAndPinningStep.
		//!
		//! An AllocationAndPinningStep fins a mapping of the host
		//! region passes through the regionTranslation on the
		//! memoryPlace and sets the _deviceStartAddress accordingly.
		//!
		//! \param[inout] when the method is called, regionTranslation
		//! 		  includes the DataAccessRegion we are allocating. The
		//!		  Step when it runs, sets up within the regionTranslation
		//!		  the device address at which the the allocation happened
		//! \param[in] memoryPlace is the MemoryPlace describing the device to
		//!	       which the allocation needs to be performed.
		Step *createAllocationAndPinningStep(
			RegionTranslation &regionTranslation,
			MemoryPlace const *memoryPlace
		);
		
		//! \brief Creates a DataCopyStep.
		//!
		//! A DataCopyStep copies (if necessary) the (host-addressed)
		//! region from the sourceMemoryPlace to the targetMemoryPlace,
		//! using the targetTranslation for the target.
		//!
		//! \param[in] sourceMemoryPlace points to the MemoryPlace from which
		//!	       need to fetch data. This can be NULL in cases in which
		//!	       the location is not known yet, e.g. the copy step
		//!	       corresponds to a weak access.
		//! \param[in] targetMemoryPlace points to the MemoryPlace to which
		//!	       need to copy data to. This cannot be NULL.
		//! \param[in] targetTranslation is a RegionTranslation which includes
		//!	       the address within the target device to which we copy data.
		//! \param[in] access is the DataAccess to which this copy step relates.
		Step *createDataCopyStep(
			MemoryPlace const *sourceMemoryPlace,
			MemoryPlace const *targetMemoryPlace,
			RegionTranslation const &targetTranslation,
			DataAccess *access
		);
		
		//! \brief Creates an ExecutionStep.
		//!
		//! An ExecutionStep executes the task on assigned computePlace.
		//!
		//! \param[in] task is the Task for which we build the execution step
		//! \param[in] computePlace is the ComputePlace on which the task will
		//!	       be executed.
		Step *createExecutionStep(
			Task *task,
			ComputePlace *computePlace
		);
		
		//! \brief Creates a NotificationStep.
		//!
		//! A NotificationStep performs the cleanup of Task after
		//! it has finished executing and notify anyone who is waiting
		//! for the Task to complete.
		//!
		//! \param[in] callback is a function to be called once the notification
		//!	       Step becomes ready.
		//! \param[in] computePlace is the ComputePlace on which the task will
		//!	       be executed.
		Step *createNotificationStep(
			std::function<void ()> const &callback,
			ComputePlace *computePlace
		);
		
		//! \brief Creates an UnpinningStep.
		//!
		//! An UnpinningStep unpins the region of the targetTranslation
		//! on the targetMemoryPlace.
		//!
		//! \param[in] targetMemoryPlace is the memoryPlace that includes the
		//!	       DataAccessRegion we are unpinning.
		//! \param[in] targetTranslation describes the DataAccessRegion and
		//! 	       its mapping in the target device.
		Step *createUnpinningStep(
			MemoryPlace const *targetMemoryPlace,
			RegionTranslation const &targetTranslation
		);
		
		//! \brief Creates a DataReleaseStep.
		//!
		//! A DataReleaseStep triggers events related to the release
		//! of regions of a DataAccess
		//!
		//! \param[in] task is the Task for which we release an access.
		//! \param[in] access is the DataAccess that we are releasing.
		Step *createDataReleaseStep(
			Task const *task,
			DataAccess *access
		);
		
		// \brief Enforces order between two steps of the Task execution.
		//
		// Create execute-after relationship between the two Steps of the workflow
		// so that the 'successor' Step will only start after the 'predecessor'
		// Step has completed.
		inline void enforceOrder(Step *predecessor, Step *successor)
		{
			if (predecessor == nullptr || successor == nullptr) {
				return;
			}
			
			predecessor->addSuccessor(successor);
			successor->addPredecessor();
		}
		
		//! \brief Add a root step to the Workflow.
		//!
		//! Root steps of the workflow are those Steps that do not have
		//! any predecessor. When the Workflow starts executing,
		//! essentially fires the execution of those Steps.
		inline void addRootStep(Step *root)
		{
			_rootSteps.push_back(root);
		}
		
		//! \brief Starts the execution of the workflow.
		//!
		//! This will start the execution of the root steps of the workflow.
		//! Whether the execution of the workflow has been completed when this
		//! method returns, depends on the nature of the steps from which the
		//! workflow consists of.
		void start();
	};
	
	// NOTE: objects of this class self-destruct when they finish
	template <typename CONTENTS_T>
	class Workflow : public WorkflowBase, CONTENTS_T {
	public:
		template <typename... TS>
		Workflow(TS &&... argsPack)
			: WorkflowBase(), CONTENTS_T(std::forward<TS>(argsPack)...)
		{
		}
	};
	
	//! \brief Creates a new workflow object that inherits
	//! from CONTENTS_T and is constructed with the argsPack
	//! parameters.
	template <typename CONTENTS_T, typename... TS>
	inline Workflow<CONTENTS_T> *createWorkflow(TS &&... argsPack)
	{
		return new Workflow<CONTENTS_T>(std::forward<TS>(argsPack)...);
	}
	
	struct TaskExecutionWorkflowData {
		std::vector<RegionTranslation> _symbolTranslations;

		TaskExecutionWorkflowData(int symbolCount)
			: _symbolTranslations(symbolCount)
		{
		}
	};
	
	//! \brief Create a workflow for executing a task
	//!
	//! \param[in] task is the Task we want to execute
	//! \param[in] targetComputePlace is the ComputePlace that the Scheduler decided
	//!            to execute the Task on.
	//! \param[in] targetMemoryPlace is the memory place that will be used for the
	//!            execution of the task, i.e. a MemoryPlace that is directly
	//!            accessible by targetComputePlace.
	void executeTask(
		Task *task,
		ComputePlace *targetComputePlace,
		MemoryPlace *targetMemoryPlace
	);
	
	//! \brief Creates a workflow for handling taskwaits
	//!
	//! \param[in] task is the Task to which the taskwait fragment belongs to
	//! \param[in] taskwaitFragment is the taskwait fragment for which we setup the workflow
	void setupTaskwaitWorkflow(Task *task, DataAccess *taskwaitFragment);
};


#endif /* __EXECUTION_WORKFLOW_HPP__ */
