/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef TASK_WAIT_HPP
#define TASK_WAIT_HPP

namespace TaskWait {

	//! \brief Block current task and wait for its childs to complete
	//!
	//! \param[in] invocationSource A representative string indicating the calling location
	//! \param[in] fromUserCode Indicates whether this function is called from user or runtime code
	void taskWait(char const *invocationSource, bool fromUserCode = false);

}

#endif // TASK_WAIT_HPP
