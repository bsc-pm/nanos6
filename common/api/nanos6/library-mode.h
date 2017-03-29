#ifndef NANOS6_LIBRARY_MODE_H
#define NANOS6_LIBRARY_MODE_H


#ifdef __cplusplus
extern "C" {
#endif


//! \brief Spawn asynchronously a function
//! 
//! \param function the function to be spawned
//! \param args a parameter that is passed to the function
//! \param completion_callback an optional function that will be called when the function finishes
//! \param completion_args a parameter that is passed to the completion callback
//! \param label an optional name for the function
void nanos_spawn_function(
	void (*function)(void *),
	void *args,
	void (*completion_callback)(void *),
	void *completion_args,
	char const *label
);


#ifdef __cplusplus
}
#endif


#endif /* NANOS6_LIBRARY_MODE_H */
