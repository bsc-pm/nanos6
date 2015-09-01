#ifndef MAIN_TASK_HPP
#define MAIN_TASK_HPP


#include "api/nanos6_rt_interface.h"
#include "tasks/Task.hpp"


namespace nanos6 {
	typedef int main_function_t(int argc, char **argv, char **envp);
	
	extern "C" struct main_task_args_block_t {
		main_function_t *_main;
		int _argc;
		char **_argv;
		char **_envp;
		
		main_task_args_block_t(main_function_t *mainFunction, int argc, char **argv, char **envp)
			: _main(mainFunction), _argc(argc), _argv(argv), _envp(envp)
		{
		}
	};
	
	extern nanos_task_info main_task_info;
}


#endif // MAIN_TASK_HPP
