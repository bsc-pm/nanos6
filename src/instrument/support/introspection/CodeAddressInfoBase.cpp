/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/


#include "CodeAddressInfoBase.hpp"

#include <cxxabi.h>

#include <sstream>


CodeAddressInfoBase::Entry CodeAddressInfoBase::_nullEntry;
std::string CodeAddressInfoBase::_unknownFunctionName = "??";
std::string CodeAddressInfoBase::_unknownSourceLocation = "??:??";


std::map<void *, CodeAddressInfoBase::Entry> CodeAddressInfoBase::_address2Entry;
std::map<void *, CodeAddressInfoBase::Entry> CodeAddressInfoBase::_returnAddress2Entry;

std::map<std::string, CodeAddressInfoBase::function_id_t> CodeAddressInfoBase::_functionName2Id;
std::map<std::string, CodeAddressInfoBase::source_location_id_t> CodeAddressInfoBase::_sourceLocation2Id;
std::vector<std::string> CodeAddressInfoBase::_mangledFunctionNames;
std::vector<std::string> CodeAddressInfoBase::_functionNames;
std::vector<std::string> CodeAddressInfoBase::_sourceLocations;


std::string CodeAddressInfoBase::demangleSymbol(std::string const &symbol)
{
	std::string result;
	
	int demangleStatus = 0;
	char *demangledName = abi::__cxa_demangle(symbol.c_str(), nullptr, 0, &demangleStatus);
	
	if ((demangledName != nullptr) && (demangleStatus == 0)) {
		result = demangledName;
	} else {
		result = symbol;
	}
	
	if (demangledName != nullptr) {
		free(demangledName);
	}
	
	return result;
}


std::string CodeAddressInfoBase::sourceToString(char const *source, int line, int column)
{
	std::ostringstream oss;
	if (source != nullptr) {
		oss << source;
		if (line > 0) {
			oss << ":" << line;
			if (column > 0) {
				oss << ":" << column;
			}
		}
		
		return oss.str();
	}
	
	return std::string();
}


CodeAddressInfoBase::InlineFrame CodeAddressInfoBase::functionAndSourceToFrame(
	std::string const &mangledFunctionName,
	std::string const &functionName,
	std::string const &sourceLocation
) {
	InlineFrame result;
	
	{
		auto it = _functionName2Id.find(functionName);
		if (it != _functionName2Id.end()) {
			result._functionId = it->second;
		} else {
			_functionName2Id[functionName] = _functionNames.size();
			result._functionId = _functionNames.size();
			_mangledFunctionNames.push_back(mangledFunctionName);
			_functionNames.push_back(functionName);
		}
	}
	
	{
		auto it = _sourceLocation2Id.find(sourceLocation);
		if (it != _sourceLocation2Id.end()) {
			result._sourceLocationId = it->second;
		} else {
			_sourceLocation2Id[sourceLocation] = _sourceLocations.size();
			result._sourceLocationId = _sourceLocations.size();
			_sourceLocations.push_back(sourceLocation);
		}
	}
	
	return result;
}
