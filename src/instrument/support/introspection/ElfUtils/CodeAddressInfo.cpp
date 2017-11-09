/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include "CodeAddressInfo.hpp"

#include <dwarf.h>
#include <elfutils/libdw.h>
#include <elfutils/libdwfl.h>

#include <cassert>
#include <iostream>

#include <unistd.h>
#include <sys/types.h>


Dwfl *CodeAddressInfo::_dwfl = nullptr;


inline std::string CodeAddressInfo::getDebugInformationEntryName(Dwarf_Die *debugInformationEntry)
{
	assert(debugInformationEntry != nullptr);
	
	Dwarf_Attribute linkageNameAttribute;
	
	// For some reason MIPS names have their own attribute type
	Dwarf_Attribute *attr = dwarf_attr_integrate(debugInformationEntry, DW_AT_MIPS_linkage_name, &linkageNameAttribute);
	
	if (attr == nullptr) {
		attr = dwarf_attr_integrate(debugInformationEntry, DW_AT_linkage_name, &linkageNameAttribute);
	}
	
	// This is useful for lambdas. In that case we get "operator()"
	if (attr == nullptr) {
		attr = dwarf_attr_integrate(debugInformationEntry, DW_AT_name, &linkageNameAttribute);
	}
	
	assert(attr == &linkageNameAttribute);
	
	char const *linkageName = dwarf_formstring(&linkageNameAttribute);
	
	if (linkageName == nullptr) {
		linkageName = dwarf_diename(debugInformationEntry);
	}
	
	if (linkageName == nullptr) {
		linkageName = "";
	}
	
	return CodeAddressInfo::demangleSymbol(linkageName);
}


void CodeAddressInfo::init()
{
	pid_t pid = getpid();
	
	static char *debugInfoPath = nullptr;
	
	static Dwfl_Callbacks dwflCallbacks = {
		dwfl_linux_proc_find_elf,
		dwfl_standard_find_debuginfo,
		nullptr,
		&debugInfoPath,
	};
	
	_dwfl = dwfl_begin(&dwflCallbacks);
	if (_dwfl == nullptr) {
		int error = dwfl_errno();
		std::cerr << "Warning: cannot get the memory map of the process: " << dwfl_errmsg(error) << std::endl;
		return;
	}
	
	int rc = dwfl_linux_proc_report(_dwfl, pid);
	if (rc != 0) {
		std::cerr << "Warning: cannot get the memory map of the process." << std::endl;
		return;
	}
}


void CodeAddressInfo::shutdown()
{
	if (_dwfl != nullptr) {
		dwfl_end(_dwfl);
	}
}


CodeAddressInfo::Entry const &CodeAddressInfo::resolveAddress(void* address)
{
	{
		auto it = _address2Entry.find(address);
		if (it != _address2Entry.end()) {
			return it->second;
		}
	}
	
	if (_dwfl == nullptr) {
		return _nullEntry;
	}
	
	Dwarf_Addr dwflAddress = (Dwarf_Addr) address;
	
	Dwfl_Module *module = dwfl_addrmodule(_dwfl, dwflAddress);
	if (module == nullptr) {
		return _nullEntry;
	}
	
	Dwarf_Addr addressBias = 0;
	Dwarf_Die *compilationUnitDebugInformationEntry = dwfl_module_addrdie(module, dwflAddress, &addressBias);
	
	assert(compilationUnitDebugInformationEntry != nullptr);
	Dwarf_Die *scopeDebugInformationEntries = nullptr;
	int scopeEntryCount = dwarf_getscopes(compilationUnitDebugInformationEntry, dwflAddress - addressBias, &scopeDebugInformationEntries);
	if (scopeEntryCount <= 0) {
		return _nullEntry;
	}
	
	// Get the name of the function
	std::string function;
	for (int scopeEntryIndex = 0; scopeEntryIndex < scopeEntryCount; scopeEntryIndex++) {
		Dwarf_Die *scopeEntry = &scopeDebugInformationEntries[scopeEntryIndex];
		int dwarfTag = dwarf_tag(scopeEntry);
		
		if (
			(dwarfTag == DW_TAG_subprogram)
			|| (dwarfTag == DW_TAG_inlined_subroutine)
			|| (dwarfTag == DW_TAG_entry_point)
		) {
			function = getDebugInformationEntryName(scopeEntry);
			break;
		}
	}
	
	// Get the source code location
	std::string sourceLine;
	{
		Dwfl_Line *dwarfLine = dwfl_module_getsrc (module, dwflAddress);
		
		Dwarf_Addr dwarfAddress = dwflAddress;
		int line = 0;
		int column = 0;
		const char *source = dwfl_lineinfo(dwarfLine, &dwarfAddress, &line, &column, nullptr, nullptr);
		
		sourceLine = sourceToString(source, line, column);
	}
	
	// Create the entry
	Entry &entry = _address2Entry[address];
	
	// Add the current function and source location
	{
		InlineFrame currentFrame = functionAndSourceToFrame(function, sourceLine);
		entry._inlinedFrames.push_back(currentFrame);
	}
	
	Dwarf_Off scopeOffset = dwarf_dieoffset(&scopeDebugInformationEntries[0]);
	Dwarf_Addr moduleBias = 0;
	Dwarf *moduleDwarf = dwfl_module_getdwarf(module, &moduleBias);
	Dwarf_Die addressDebugInformationEntry;
	dwarf_offdie(moduleDwarf, scopeOffset, &addressDebugInformationEntry);
	free(scopeDebugInformationEntries);
	
	scopeDebugInformationEntries = nullptr;
	scopeEntryCount = dwarf_getscopes_die(&addressDebugInformationEntry, &scopeDebugInformationEntries);
	
	// May correspond to more than one function due to inlining
	if (scopeEntryCount > 1) {
		Dwarf_Die compilationUnit;
		Dwarf_Die *cu = dwarf_diecu(&scopeDebugInformationEntries[0], &compilationUnit, nullptr, nullptr);
		
		Dwarf_Files *sourceFiles = nullptr;
		
		int rc = -1;
		if (cu != nullptr) {
			assert(cu == &compilationUnit);
			
			// Load the source file information of the compilation unit if not already loaded
			rc = dwarf_getsrcfiles(&compilationUnit, &sourceFiles, nullptr);
		}
		
		if (rc == 0) {
			for (int scopeEntryIndex = 0; scopeEntryIndex < scopeEntryCount-1; scopeEntryIndex++) {
				Dwarf_Die *scopeEntry = &scopeDebugInformationEntries[scopeEntryIndex];
				{
					int dwarfTag = dwarf_tag(scopeEntry);
					
					if (dwarfTag != DW_TAG_inlined_subroutine) {
						continue;
					}
				}
				
				function.clear();
				sourceLine.clear();
				
				// Look up the function name
				for (int parentEntryIndex = scopeEntryIndex + 1; parentEntryIndex < scopeEntryCount; parentEntryIndex++) {
					Dwarf_Die *parentEntry = &scopeDebugInformationEntries[parentEntryIndex];
					int dwarfTag = dwarf_tag(parentEntry);
					
					if (
						(dwarfTag == DW_TAG_subprogram)
						|| (dwarfTag == DW_TAG_inlined_subroutine)
						|| (dwarfTag == DW_TAG_entry_point)
					) {
						function = getDebugInformationEntryName(parentEntry);
						break;
					}
				}
				
				// Get the source code location
				{
					int line = 0;
					int column = 0;
					const char *source = nullptr;
					
					Dwarf_Word attributeValue;
					Dwarf_Attribute attribute;
					
					// Get source file
					Dwarf_Attribute *filledAttribute = dwarf_attr(scopeEntry, DW_AT_call_file, &attribute);
					if (filledAttribute != nullptr) {
						assert(filledAttribute == &attribute);
						rc = dwarf_formudata(&attribute, &attributeValue);
						if (rc == 0) {
							source = dwarf_filesrc(sourceFiles, attributeValue, nullptr, nullptr);
						}
					}
					
					// Get line number
					filledAttribute = dwarf_attr(scopeEntry, DW_AT_call_line, &attribute);
					if (filledAttribute != nullptr) {
						assert(filledAttribute == &attribute);
						rc = dwarf_formudata(&attribute, &attributeValue);
						if (rc == 0) {
							line = attributeValue;
						}
					}
					
					// Get column number
					filledAttribute = dwarf_attr(scopeEntry, DW_AT_call_column, &attribute);
					if (filledAttribute != nullptr) {
						assert(filledAttribute == &attribute);
						rc = dwarf_formudata(&attribute, &attributeValue);
						if (rc == 0) {
							column = attributeValue;
						}
					}
					
					sourceLine = sourceToString(source, line, column);
				}
				
				// Add the current function and source location
				InlineFrame currentFrame = functionAndSourceToFrame(function, sourceLine);
				entry._inlinedFrames.push_back(currentFrame);
			}
		}
		
		free(scopeDebugInformationEntries);
	}
	
	return entry;
}
