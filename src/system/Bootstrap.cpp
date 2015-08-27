#include <iostream>

#include <assert.h>
#include <dlfcn.h>

#include "LeaderThread.hpp"
#include "MainTask.hpp"

#include "executors/threads/ThreadManager.hpp"
#include "executors/threads/ThreadManagerPolicy.hpp"
#include "scheduling/Scheduler.hpp"


// The actual "main" function that the runtime will wrap as a task
static MainTask::main_function_t *appMain;


#if __powerpc__
struct startup_info {
	void *sda_base;
	int (*main) (int, char **, char **);
	void (*init) (void);
	void (*fini) (void);
};

extern "C" typedef int libc_start_main_function_t(
	int argc,
	char **argv,
	char **envp,
	void *auxvec,
	void (*rtld_fini) (void),
	struct startup_info *startupInfo,
	char **stackOnEntry
);
#else
extern "C" typedef int libc_start_main_function_t(
	int (*main) (int, char **, char **),
	int argc,
	char **argv,
	void (*init) (void),
	void (*fini) (void),
	void (*rtld_fini) (void),
	void *stack_end
);
#endif


namespace simpless {
	//! \brief This function creates the main task, submits it and keeps the initial thread doing maintainance duties
	//!
	//! \returns the return code of the "main" function
	static int bootstrap(int argc, char **argv, char **envp) {
		Scheduler::initialize();
		ThreadManagerPolicy::initialize();
		
		Task *mainTask = new MainTask(appMain, argc, argv, envp);
		Scheduler::addReadyTask(mainTask, nullptr);
		
		ThreadManager::initialize();
		
		LeaderThread::maintenanceLoop();
		
		ThreadManager::shutdown();
		
		return LeaderThread::getMainReturnCode();
	}
}


//! \brief This function overrides the function of the same name and is in charge of handling the call to "main". In this case, it calls simpless::bootstrap instead.
#if __powerpc__
extern "C" int __libc_start_main(
	int argc,
	char **argv,
	char **envp,
	void *auxvec,
	void (*rtld_fini) (void),
	struct startup_info *startupInfo,
	char **stackOnEntry
) {
	// Find the actual function that we are overriding
	libc_start_main_function_t * real_libc_start_main =
		(libc_start_main_function_t *) dlsym(RTLD_NEXT, "__libc_start_main");
	assert(real_libc_start_main != nullptr);
	
	// Interpose simpless::bootstrap instead of main into the overrided function
	assert(startupInfo != nullptr);
	appMain = startupInfo->main;
	
	struct startup_info newStartupInfo = { startupInfo->sda_base, simpless::bootstrap, startupInfo->init, startupInfo->fini };
	
	return real_libc_start_main(argc, argv, envp, auxvec, rtld_fini, &newStartupInfo, stackOnEntry);
}
#else
extern "C" int __libc_start_main(
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
	return real_libc_start_main(simpless::bootstrap, argc, argv, init, fini, rtld_fini, stack_end);
}
#endif
