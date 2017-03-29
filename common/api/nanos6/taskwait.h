#ifndef NANOS6_TASKWAIT_H
#define NANOS6_TASKWAIT_H


#ifdef __cplusplus
extern "C" {
#endif


//! \brief Block the control flow of the current task until all of its children have finished
//!
//! \param[in] invocation_source A string that identifies the source code location of the invocation
void nanos_taskwait(char const *invocation_source);


#ifdef __cplusplus
}
#endif


#endif // NANOS6_TASKWAIT_H
