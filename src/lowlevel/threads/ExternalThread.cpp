/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2017 Barcelona Supercomputing Center (BSC)
*/


#include "ExternalThread.hpp"


__thread ExternalThread *ExternalThread::_currentExternalThread = nullptr;

