#ifndef MAIN_TASK_HPP
#define MAIN_TASK_HPP


#include "tasks/Task.hpp"


class MainTask: public Task {
public:
	typedef int main_function_t(int argc, char **argv, char **envp);
	
private:
	main_function_t *_main;
	int _argc;
	char **_argv;
	char **_envp;
	
public:
	MainTask(main_function_t *mainFunction, int argc, char **argv, char **envp)
		: Task(nullptr),
		_main(mainFunction), _argc(argc), _argv(argv), _envp(envp)
	{
		
	}
	
	virtual ~MainTask()
	{
		
	}
	
	
	void body();
};


#endif // MAIN_TASK_HPP
