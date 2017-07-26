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
		: _retriever(nullptr), _enabledByDefault(false)
	{
	}
	
	OptionHelper(
		std::string &&parameter, std::string &&helpMessage,
		std::string &&header, char const * (*retriever)(), bool enabledByDefault = false
	)
		: _parameter(std::move(parameter)), _helpMessage(std::move(helpMessage)),
		_header(std::move(header)), _retriever(retriever),
		_enabledByDefault(enabledByDefault)
	{
	}
	
	bool empty() const
	{
		return (_retriever == nullptr);
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
	for (auto const &optionHelper : optionHelpers) {
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
	char const *branch = nanos_get_runtime_branch();
	std::cout << "Nanos6 version " << nanos_get_runtime_version();
	if ((branch != nullptr) && (std::string(branch) != "master")) {
		std::cout << " " << branch << " branch";
	}
	
	char const *patches = nanos_get_runtime_patches();
	if ((patches != nullptr) && (std::string() != patches)) {
		std::cout << " +changes";
	}
	
	std::cout << std::endl;
	
	return "";
}


static char const *dumpPatches()
{
	char const *patches = nanos_get_runtime_patches();
	
	if (patches == nullptr) {
		std::cerr << "Error: this is either a runtime compiled from a distributed tarball or it has been compiled with code change reporting disabled." << std::endl;
		std::cerr << "To enable code change reporting, please configure the runtime with the --enable-embed-code-changes parameter." << std::endl;
		
		exit(1);
	}
	
	if (std::string() != patches) {
		std::cout << patches;
	} else {
		std::cout << "This runtime does not contain any changes over the reported version." << std::endl;
	}
}


static char const *dumpRuntimeDetailedInfo()
{
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
	
	char const *patches = nanos_get_runtime_patches();
	if ((patches != nullptr) && (std::string() != patches)) {
		std::cout << "This runtime contains patches" << std::endl;
	}
}



int main(int argc, char **argv)
{
	commandName = argv[0];
	
	optionHelpers.emplace_back("--help", "display this help message", "", emitHelp);
	optionHelpers.emplace_back();
	optionHelpers.emplace_back("--full-version", "display the full runtime version", "", showFullVersion, true);
	optionHelpers.emplace_back("--copyright", "display the copyright notice", "Copyright (C)", nanos_get_runtime_copyright, true);
	optionHelpers.emplace_back();
	optionHelpers.emplace_back("--version", "display the runtime version", "Nanos6 version", nanos_get_runtime_version);
	optionHelpers.emplace_back("--branch", "display the runtime branch", "Nanos6 branch", nanos_get_runtime_branch);
	optionHelpers.emplace_back();
	optionHelpers.emplace_back("--runtime-compiler", "display the compiler used for this runtime", "Compiled with", nanos_get_runtime_compiler_version);
	optionHelpers.emplace_back("--runtime-compiler-flags", "display the compiler flags used for this runtime", "Compilation flags", nanos_get_runtime_compiler_flags);
	optionHelpers.emplace_back();
	optionHelpers.emplace_back("--runtime-details", "display detailed runtime and execution environment information", "", dumpRuntimeDetailedInfo);
	optionHelpers.emplace_back("--dump-patches", "display code changes over the reported version", "", dumpPatches);
	
	if (argc > 1) {
		for (int i = 1; i < argc; i++) {
			auto it = optionHelpers.begin();
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
		for (auto const &optionHelper : optionHelpers) {
			if (optionHelper._enabledByDefault) {
				optionHelper.emit();
			}
		}
	}
	
	return 0;
}

