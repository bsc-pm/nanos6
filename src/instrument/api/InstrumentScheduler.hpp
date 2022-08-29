/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020-2022 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_SCHEDULER_SUBSYTEM_ENTRY_POINTS_HPP
#define INSTRUMENT_SCHEDULER_SUBSYTEM_ENTRY_POINTS_HPP

#include "tasks/Task.hpp"

namespace Instrument {

	//! \brief Enters the scheduler addReadyTask method
	void enterAddReadyTask();

	//! \brief Exits the scheduler addReadyTask method
	void exitAddReadyTask();

	//! \brief Enters the scheduler getReadyTask method
	void enterGetReadyTask();

	//! \brief Exits the scheduler getReadyTask method
	void exitGetReadyTask(Task *task);

	//! \brief Enters the scheduler processReadyTasks method
	void enterProcessReadyTasks();

	//! \brief Exits the scheduler processReadyTasks method
	void exitProcessReadyTasks();

	//! \brief The current worker enters the scheduler subscription lock
	void enterSchedulerLock();

	//! \brief The current worker becomes a server. It will serve tasks to other workers
	void schedulerLockBecomesServer();

	//! \brief The current worker exits the subscription lock busy wait loop after having received a task from another worker
	//! \param[in] taskId the identifier of the task this client has been assigned
	void exitSchedulerLockAsClient(task_id_t taskId);

	//! \brief The current worker exits the subscription lock busy wait loop without having received a task
	void exitSchedulerLockAsClient();

	//! \brief The current thread serves a task to another thread
	//! \param[in] taskId the identifier of the task that the server assigned to the client
	void schedulerLockServesTask(task_id_t taskId);

	//! \brief The current worker exits the subscription lock after having had the "server" role.
	void exitSchedulerLockAsServer();

	//! \brief The current worker exits the subscription lock after having had the "server" role.
	//! \param[in] taskId the identifier of the task that the server assigned itself
	void exitSchedulerLockAsServer(task_id_t taskId);

}

#endif // INSTRUMENT_SCHEDULER_SUBSYTEM_ENTRY_POINTS_HPP
