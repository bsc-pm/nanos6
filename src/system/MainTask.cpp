#include "api/nanos6_rt_interface.h"
#include "MainTask.hpp"
#include "LeaderThread.hpp"


namespace nanos6 {
	extern "C" {
		static void main_task_wrapper(void *argsBlock)
		{
			main_task_args_block_t *realArgsBlock = (main_task_args_block_t *)argsBlock;
			
			assert(realArgsBlock != nullptr);
			assert(realArgsBlock->_main != nullptr);
			
			int returnCode = realArgsBlock->_main(
				realArgsBlock->_argc,
				realArgsBlock->_argv,
				realArgsBlock->_envp
			);
			nanos_taskwait("Nanos6-Bootstrap-Code");
			
			LeaderThread::notifyMainExit(returnCode);
		}
		
		
		static void main_task_register_depinfo(__attribute__((unused)) void *handler, __attribute__((unused)) void *argsBlock)
		{
		}
		
		
		static void main_task_register_copies(__attribute__((unused)) void *handler, __attribute__((unused)) void *argsBlock)
		{
		}
	}
	
	nanos_task_info main_task_info = {
		main_task_wrapper,
		main_task_register_depinfo,
		main_task_register_copies,
		"main",
		""
	};
	
	nanos_task_invocation_info main_task_invocation_info = {
		"Nanos6-Bootstrap-Code"
	};
	
}


