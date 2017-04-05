#include <assert.h>
#include <stddef.h>

#include "intercept-main-common.h"
#include "main-wrapper.h"
#include "loader.h"


typedef struct {
	void (**preinit_array)(void);
	void (**init_array)(void);
	void (**fini_array)(void);
} structors_array_t;


typedef void libc_start_main_function_t(
	void *raw_args,
	void (*onexit)(void),
	int (*slingshot)(int, char **, char **),
	structors_array_t const * const structors
);


__attribute__ ((visibility ("hidden")))  libc_start_main_function_t *_nanos6_loader_next_libc_start_main = NULL;


//! \brief This function overrides the function of the same name and is in charge of loading the Nanos6 runtime
void __libc_init(
	void *raw_args,
	void (*onexit)(void),
	int (*slingshot)(int, char **, char **),
	structors_array_t const * const structors
) {
	_nanos6_resolve_next_start_main("__libc_init");
	
	assert(_nanos6_loader_wrapped_main == 0);
	_nanos6_loader_wrapped_main = slingshot;
	
	// Continue with the "normal" startup sequence
	_nanos6_loader_next_libc_start_main(raw_args, onexit, _nanos6_loader_main, structors);
}

