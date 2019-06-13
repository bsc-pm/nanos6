/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include <nanos6/bootstrap.h>
#include <nanos6/debug.h>
#include <nanos6/library-mode.h>
#include <nanos6/runtime-info.h>

#include <cstdlib>
#include <iostream>
#include <list>
#include <string>
#include <utility>


struct OptionHelper {
	enum retriever_t {
		command_help = 0,
		runtime_branch,
		runtime_compiler_version,
		runtime_compiler_flags,
		runtime_copyright,
		runtime_detailed_info,
		runtime_full_license,
		runtime_full_version,
		runtime_license,
		runtime_patches,
		runtime_path,
		runtime_version,
		no_retriever
	};

	std::string _parameter;
	std::string _helpMessage;
	std::string _header;
	retriever_t _retriever;
	bool _enabledByDefault;
	
	OptionHelper()
		: _retriever(OptionHelper::no_retriever), _enabledByDefault(false)
	{
	}
	
	OptionHelper(
		std::string &parameter, std::string &helpMessage,
		std::string &header, retriever_t retriever, bool enabledByDefault = false
	)
		: _parameter(parameter), _helpMessage(helpMessage),
		_header(header), _retriever(retriever),
		_enabledByDefault(enabledByDefault)
	{
	}
	
	OptionHelper(
		char const *parameter, char const *helpMessage,
		char const *header, retriever_t retriever, bool enabledByDefault = false
	)
		: _parameter(parameter), _helpMessage(helpMessage),
		_header(header), _retriever(retriever),
		_enabledByDefault(enabledByDefault)
	{
	}
	
	bool empty() const
	{
		return (_retriever == OptionHelper::no_retriever);
	}
	
	void emit() const
	{
		if (!empty()) {
			if (_header != "") {
				std::cout << _header << " " << retrieve(_retriever) << std::endl;
			} else {
				retrieve(_retriever);
			}
		}
	}
	
	static char const *retrieve(retriever_t retriever);
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

char const *OptionHelper::retrieve(retriever_t retriever)
{
	switch (retriever) {
		case command_help:
			return emitHelp();
		case runtime_branch:
			return nanos6_get_runtime_branch();
		case runtime_compiler_version:
			return nanos6_get_runtime_compiler_version();
		case runtime_compiler_flags:
			return nanos6_get_runtime_compiler_flags();
		case runtime_copyright:
			return nanos6_get_runtime_copyright();
		case runtime_detailed_info:
			return dumpRuntimeDetailedInfo();
		case runtime_full_license:
			return nanos6_get_runtime_full_license();
		case runtime_full_version:
			return showFullVersion();
		case runtime_license:
			return nanos6_get_runtime_license();
		case runtime_patches:
			return dumpPatches();
		case runtime_path:
			return nanos6_get_runtime_path();
		case runtime_version:
			return nanos6_get_runtime_version();
		default:
			abort();
	}
}


int main(int argc, char **argv)
{
	// Runtime initialization
	char const *error = nanos6_library_mode_init();
	if (error != NULL) {
		std::cerr << "Error initializing Nanos6 runtime system: " << error << std::endl;
		return 1;
	}
	
	commandName = argv[0];
	
	optionHelpers.push_back(OptionHelper("--help", "display this help message", "", OptionHelper::command_help));
	optionHelpers.push_back(OptionHelper());
	optionHelpers.push_back(OptionHelper("--full-version", "display the full runtime version", "", OptionHelper::runtime_full_version, true));
	optionHelpers.push_back(OptionHelper("--copyright", "display the copyright notice", "Copyright (C)", OptionHelper::runtime_copyright, true));
	optionHelpers.push_back(OptionHelper());
	optionHelpers.push_back(OptionHelper("--license", "display the license type", "Licensed as", OptionHelper::runtime_license, true));
	optionHelpers.push_back(OptionHelper("--full-license", "display the license terms", "Licensing terms:\n", OptionHelper::runtime_full_license));
	optionHelpers.push_back(OptionHelper());
	optionHelpers.push_back(OptionHelper("--version", "display the runtime version", "Nanos6 version", OptionHelper::runtime_version));
	optionHelpers.push_back(OptionHelper("--branch", "display the runtime branch", "Nanos6 branch", OptionHelper::runtime_branch));
	optionHelpers.push_back(OptionHelper());
	optionHelpers.push_back(OptionHelper("--runtime-compiler", "display the compiler used for this runtime", "Compiled with", OptionHelper::runtime_compiler_version));
	optionHelpers.push_back(OptionHelper("--runtime-compiler-flags", "display the compiler flags used for this runtime", "Compilation flags", OptionHelper::runtime_compiler_flags));
	optionHelpers.push_back(OptionHelper("--runtime-path", "display the path of the loaded runtime", "Runtime path", OptionHelper::runtime_path));
	optionHelpers.push_back(OptionHelper());
	optionHelpers.push_back(OptionHelper("--runtime-details", "display detailed runtime and execution environment information", "", OptionHelper::runtime_detailed_info));
	optionHelpers.push_back(OptionHelper("--dump-patches", "display code changes over the reported version", "", OptionHelper::runtime_patches));
	
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
	
	// Runtime shutdown
	nanos6_shutdown();
	
	return 0;
}

