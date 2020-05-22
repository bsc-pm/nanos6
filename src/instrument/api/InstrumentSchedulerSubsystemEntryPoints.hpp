/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_SCHEDULER_SUBSYTEM_ENTRY_POINTS_HPP
#define INSTRUMENT_SCHEDULER_SUBSYTEM_ENTRY_POINTS_HPP


namespace Instrument {

	//! \brief Enters the scheduler addReadyTask method
	void enterAddReadyTask();

	//! \brief Exits the scheduler addReadyTask method
	void exitAddReadyTask();

	//! \brief Enters the scheduler addReadyTask method
	void enterGetReadyTask();

	//! \brief Exits the scheduler addReadyTask method
	void exitGetReadyTask();

}

#endif // INSTRUMENT_SCHEDULER_SUBSYTEM_ENTRY_POINTS_HPP
