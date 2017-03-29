#ifndef NANOS6_BOOTSTRAP_H
#define NANOS6_BOOTSTRAP_H


#ifdef __cplusplus
extern "C" {
#endif


//! \brief Initialize the runtime at least to the point that it will accept tasks
void nanos_preinit(void);

//! \brief Continue with the rest of the runtime initialization
void nanos_init(void);

//! \brief Force the runtime to be shut down
// 
// This function is used to shut down the runtime
void nanos_shutdown(void);


#ifdef __cplusplus
}
#endif


#endif /* NANOS6_BOOTSTRAP_H */
