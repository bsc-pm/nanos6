#ifndef NANOS6_LOADER_MAIN_WRAPPER_H
#define NANOS6_LOADER_MAIN_WRAPPER_H


typedef int main_function_t(int argc, char **argv, char **envp);

extern main_function_t *_nanos6_loader_wrapped_main;
extern main_function_t *main;
main_function_t _nanos6_loader_main;


#endif /* NANOS6_LOADER_MAIN_WRAPPER_H */
