#include "MainTask.hpp"
#include "LeaderThread.hpp"
#include "system/ompss/TaskWait.hpp"


void MainTask::body()
{
	int returnCode = _main(_argc, _argv, _envp);
	ompss::taskWait();
	LeaderThread::notifyMainExit(returnCode);
}

