#ifndef EXECUTION_WORKFLOW_HPP
#define EXECUTION_WORKFLOW_HPP

#include "ExecutionStep.hpp"

class Task;
class ComputePlace;
class MemoryPlace;

namespace ExecutionWorkflow {
	
	class WorkflowBase {
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

		TaskExecutionWorkflowData(__attribute__((unused))int symbolCount)
		{
		}
	};
	
	void executeTask(
		Task *task,
		ComputePlace *targetComputePlace,
		MemoryPlace *targetMemoryPlace
	);
	
	void setupTaskwaitWorkflow(
		Task *task,
		DataAccess *taskwaitFragment
	);
}

#endif /* EXECUTION_WORKFLOW_HPP */
