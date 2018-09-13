/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include <nanos6/debug.h>
#include <nanos6/runtime-info.h>

#include <cstdlib>
#include <iostream>
#include <list>
#include <string>
#include <utility>


struct OptionHelper {
	std::string _parameter;
	std::string _helpMessage;
	std::string _header;
	char const * (*_retriever)();
	bool _enabledByDefault;
	
	OptionHelper()
		: _retriever(0), _enabledByDefault(false)
	{
	}
	
	OptionHelper(
		std::string &parameter, std::string &helpMessage,
		std::string &header, char const * (*retriever)(), bool enabledByDefault = false
	)
		: _parameter(parameter), _helpMessage(helpMessage),
		_header(header), _retriever(retriever),
		_enabledByDefault(enabledByDefault)
	{
	}
	
	OptionHelper(
		char const *parameter, char const *helpMessage,
		char const *header, char const * (*retriever)(), bool enabledByDefault = false
	)
		: _parameter(parameter), _helpMessage(helpMessage),
		_header(header), _retriever(retriever),
		_enabledByDefault(enabledByDefault)
	{
	}
	
	bool empty() const
	{
		return (_retriever == 0);
	}
	
	void emit() const
	{
		if (!empty()) {
			if (_header != "") {
				std::cout << _header << " " << _retriever() << std::endl;
			} else {
				_retriever();
			}
		}
	}
};


static std::string commandName;
static std::list<OptionHelper> optionHelpers;


static char const *emitHelp()
{
	std::cout << "Usage: " << commandName << " <options>" << std::endl;
	std::cout << std::endl;
	std::cout << "Options:" << std::endl;
	for (std::list<OptionHelper>::const_iterator it = optionHelpers.begin(); it != optionHelpers.end(); it++) {
		OptionHelper const &optionHelper = *it;
		if (!optionHelper.empty()) {
			std::cout << "\t" << optionHelper._parameter << "\t" << optionHelper._helpMessage << std::endl;
		} else {
			std::cout << std::endl;
		}
	}
	std::cout << std::endl;
	
	return "";
}


static char const *showFullVersion()
{
	char const *branch = nanos6_get_runtime_branch();
	std::cout << "Nanos6 version " << nanos6_get_runtime_version();
	if ((branch != 0) && (std::string(branch) != "master")) {
		std::cout << " " << branch << " branch";
	}
	
	char const *patches = nanos6_get_runtime_patches();
	if ((patches != 0) && (std::string() != patches)) {
		std::cout << " +changes";
	}
	
	std::cout << std::endl;
	
	return "";
}


static char const *dumpPatches()
{
	char const *patches = nanos6_get_runtime_patches();
	
	if (patches == 0) {
		std::cerr << "Error: this is either a runtime compiled from a distributed tarball or it has been compiled with code change reporting disabled." << std::endl;
		std::cerr << "To enable code change reporting, please configure the runtime with the --enable-embed-code-changes parameter." << std::endl;
		
		exit(1);
	}
	
	if (std::string() != patches) {
		std::cout << patches;
	} else {
		std::cout << "This runtime does not contain any changes over the reported version." << std::endl;
	}
	
	return "";
}


static char const *dumpRuntimeDetailedInfo()
{
	std::cout << "Runtime path " << nanos6_get_runtime_path() << std::endl;
	
	for (void *it = nanos6_runtime_info_begin(); it != nanos6_runtime_info_end(); it = nanos6_runtime_info_advance(it)) {
		nanos6_runtime_info_entry_t entry;
		nanos6_runtime_info_get(it, &entry);
		
		std::cout << entry.description << " ";
		switch (entry.type) {
			case nanos6_integer_runtime_info_entry:
				std::cout << entry.integer;
				break;
			case nanos6_real_runtime_info_entry:
				std::cout << entry.real;
				break;
			case nanos6_text_runtime_info_entry:
				std::cout << entry.text;
				break;
		}
		
		if (std::string() != entry.units) {
			std::cout << " " << entry.units;
		}
		
		std::cout << std::endl;
	}
	
	char const *patches = nanos6_get_runtime_patches();
	if ((patches != 0) && (std::string() != patches)) {
		std::cout << "This runtime contains patches" << std::endl;
	}
	
	return "";
}



int main(int argc, char **argv)
{
	commandName = argv[0];
	
	optionHelpers.push_back(OptionHelper("--help", "display this help message", "", emitHelp));
	optionHelpers.push_back(OptionHelper());
	optionHelpers.push_back(OptionHelper("--full-version", "display the full runtime version", "", showFullVersion, true));
	optionHelpers.push_back(OptionHelper("--copyright", "display the copyright notice", "Copyright (C)", nanos6_get_runtime_copyright, true));
	optionHelpers.push_back(OptionHelper());
	optionHelpers.push_back(OptionHelper("--license", "display the license type", "Licensed as", nanos6_get_runtime_license, true));
	optionHelpers.push_back(OptionHelper("--full-license", "display the license terms", "Licensing terms:\n", nanos6_get_runtime_full_license));
	optionHelpers.push_back(OptionHelper());
	optionHelpers.push_back(OptionHelper("--version", "display the runtime version", "Nanos6 version", nanos6_get_runtime_version));
	optionHelpers.push_back(OptionHelper("--branch", "display the runtime branch", "Nanos6 branch", nanos6_get_runtime_branch));
	optionHelpers.push_back(OptionHelper());
	optionHelpers.push_back(OptionHelper("--runtime-compiler", "display the compiler used for this runtime", "Compiled with", nanos6_get_runtime_compiler_version));
	optionHelpers.push_back(OptionHelper("--runtime-compiler-flags", "display the compiler flags used for this runtime", "Compilation flags", nanos6_get_runtime_compiler_flags));
	optionHelpers.push_back(OptionHelper("--runtime-path", "display the path of the loaded runtime", "Runtime path", nanos6_get_runtime_path));
	optionHelpers.push_back(OptionHelper());
	optionHelpers.push_back(OptionHelper("--runtime-details", "display detailed runtime and execution environment information", "", dumpRuntimeDetailedInfo));
	optionHelpers.push_back(OptionHelper("--dump-patches", "display code changes over the reported version", "", dumpPatches));
	
	if (argc > 1) {
		for (int i = 1; i < argc; i++) {
			std::list<OptionHelper>::const_iterator it = optionHelpers.begin();
			for (; it != optionHelpers.end(); it++) {
				if (it->_parameter == argv[i]) {
					break;
				}
			}
			if (it == optionHelpers.end()) {
				std::cerr << "Unknown option " << argv[i] << std::endl;
				emitHelp();
				exit(1);
			} else {
				it->emit();
			}
		}
	} else {
		// Default output
		for (std::list<OptionHelper>::const_iterator it = optionHelpers.begin(); it != optionHelpers.end(); it++) {
			OptionHelper const &optionHelper = *it;
			if (optionHelper._enabledByDefault) {
				optionHelper.emit();
			}
		}
	}
	
	return 0;
}

