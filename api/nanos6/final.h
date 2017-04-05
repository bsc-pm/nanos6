#ifndef NANOS6_FINAL_H
#define NANOS6_FINAL_H

enum nanos6_final_api_t { nanos6_final_api = 1 };


#ifdef __cplusplus
extern "C" {
#endif


//! \brief Check if running in a final context
signed int nanos_in_final(void);


#ifdef __cplusplus
}
#endif


#endif /* NANOS6_FINAL_H */
