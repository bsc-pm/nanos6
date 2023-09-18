/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020-2022 Barcelona Supercomputing Center (BSC)
*/

#ifndef SYMBOL_TRANSLATION_HPP
#define SYMBOL_TRANSLATION_HPP

#include <nanos6.h>

#include "tasks/Task.hpp"

#include <DataAccessRegistration.hpp>
#include <MemoryAllocator.hpp>

class ComputePlace;

class SymbolTranslation {
public:
	// Constexpr because we want to force the compiler to not generate a VLA
	static constexpr int MAX_STACK_SYMBOLS = 20;

	static inline nanos6_address_translation_entry_t *generateTranslationTable(
		Task *task,
		ComputePlace *computePlace,
		nanos6_address_translation_entry_t *stackTable,
		/* output */ size_t &tableSize
	) {
		nanos6_address_translation_entry_t *table = nullptr;
		nanos6_task_info_t const *const taskInfo = task->getTaskInfo();
		int numSymbols = taskInfo->num_symbols;
		if (numSymbols == 0)
			return nullptr;

		// Use stack-allocated table if there are just a few symbols, to prevent extra allocations
		if (numSymbols <= MAX_STACK_SYMBOLS) {
			tableSize = 0;
			table = stackTable;
		} else {
			tableSize = numSymbols * sizeof(nanos6_address_translation_entry_t);
			table = (nanos6_address_translation_entry_t *)
				MemoryAllocator::alloc(tableSize);
		}

		// Initialize translationTable
		for (int i = 0; i < numSymbols; ++i)
			table[i] = {0, 0};

		return table;
	}

	static inline void translateReductions(
		Task *task,
		ComputePlace *computePlace,
		nanos6_address_translation_entry_t *table,
		int numSymbols
	) {
		DataAccessRegistration::translateReductionAddresses(task, computePlace, table, numSymbols);
	}
};

#endif // SYMBOL_TRANSLATION_HPP
