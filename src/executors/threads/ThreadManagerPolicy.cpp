#include "DefaultThreadManagerPolicy.hpp"
#include "ThreadManagerPolicy.hpp"


ThreadManagerPolicyInterface *ThreadManagerPolicy::_policy = nullptr;


void ThreadManagerPolicy::initialize()
{
	_policy = new DefaultThreadManagerPolicy();
}


#include "executors/threads/WorkerThread.hpp"
#include "executors/threads/WorkerThreadImplementation.hpp"
