#include <cassert>
#include <cstdlib>
#include <dlfcn.h>

#include "ProgramLifecycle.hpp"
#include "TestAnyProtocolProducer.hpp"
#include "Timer.hpp"


//! \brief Tests should define this function that will be called after the runtime has shut down
extern void shutdownTests();


TestAnyProtocolProducer tap;
Timer initializationTimer;
Timer shutdownTimer;


int (*next_main) (int, char **, char **);


int test_main(int argc, char **argv, char **envp)
{
	initializationTimer.start();
	
	int rc = next_main(argc, argv, envp);
	
	if (shutdownTimer.hasBeenStartedAtLeastOnce()) {
		shutdownTimer.stop();
	}
	
	if (initializationTimer.hasBeenStoppedAtLeastOnce()) {
		tap.emitDiagnostic("Startup time: ", (long int) initializationTimer, " us");
	}
	if (shutdownTimer.hasBeenStoppedAtLeastOnce()) {
		tap.emitDiagnostic("Shutdown time: ", (long int) shutdownTimer, " us");
	}
	
	shutdownTests();
	return rc;
}


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
	
	assert(startupInfo != nullptr);
	next_main = startupInfo->main;
	
	struct startup_info newStartupInfo = { startupInfo->sda_base, test_main, startupInfo->init, startupInfo->fini };
	
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
	
	next_main = main;
	
	return real_libc_start_main(test_main, argc, argv, init, fini, rtld_fini, stack_end);
}
#endif

