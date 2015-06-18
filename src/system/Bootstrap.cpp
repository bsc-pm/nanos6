#include <iostream>

#include <assert.h>
#include <dlfcn.h>

#include "LeaderThread.hpp"
#include "MainTask.hpp"

#include "executors/threads/ThreadManager.hpp"
#include "scheduling/Scheduler.hpp"


// The actual "main" function that the runtime will wrap as a task
static MainTask::main_function_t *appMain;


extern "C" typedef void libc_start_main_function_t(
	int (*main) (int, char **, char **),
	int argc,
	char **argv,
	void (*init) (void),
	void (*fini) (void),
	void (*rtld_fini) (void),
	void *stack_end
);


namespace simpless {
	//! \brief This function creates the main task, submits it and keeps the initial thread doing maintainance duties
	//!
	//! \returns the return code of the "main" function
	static int bootstrap(int argc, char **argv, char **envp) {
		Scheduler::initialize();
		
		Task *mainTask = new MainTask(appMain, argc, argv, envp);
		Scheduler::addMainTask(mainTask);
		
		ThreadManager::initialize();
		
		LeaderThread::maintenanceLoop();
		
		ThreadManager::shutdown();
		
		return LeaderThread::getMainReturnCode();
	}
}


//! \brief This function overrides the function of the same name and is in charge of handling the call to "main". In this case, it calls simpless::bootstrap instead.
extern "C" void __libc_start_main(
	int (*main) (int, char **, char **),
	int argc,
	char **argv,
	void (*init) (void),
	void (*fini) (void),
	void (*rtld_fini) (void),
	void *stack_end
) {
	// Find the actual function that we are overriding
	libc_start_main_function_t * real_libc_start_main =
		(libc_start_main_function_t *) dlsym(RTLD_NEXT, "__libc_start_main");
	assert(real_libc_start_main != nullptr);
	
	appMain = main;
	
	// Interpose simpless::bootstrap instead of main into the overrided function
	real_libc_start_main(simpless::bootstrap, argc, argv, init, fini, rtld_fini, stack_end);
}

