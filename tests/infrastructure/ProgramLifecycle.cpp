#include <cassert>
#include <cstdlib>
#include <dlfcn.h>

#include "TestAnyProtocolProducer.hpp"


//! \brief Tests should define this function that will be called after the runtime has shut down
extern void shutdownTests();


TestAnyProtocolProducer tap;


int (*next_main) (int, char **, char **);


int test_main(int argc, char **argv, char **envp)
{
	int rc = next_main(argc, argv, envp);
	shutdownTests();
	return rc;
}


extern "C" typedef void libc_start_main_function_t(
	int (*main) (int, char **, char **),
	int argc,
	char **argv,
	void (*init) (void),
	void (*fini) (void),
	void (*rtld_fini) (void),
	void *stack_end
);


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
	
	next_main = main;
	
	real_libc_start_main(test_main, argc, argv, init, fini, rtld_fini, stack_end);
}

