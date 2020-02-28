/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef DATA_ACCESS_FLAGS_HPP
#define DATA_ACCESS_FLAGS_HPP

#include <stack>

struct DataAccess;

//! Definitions for the access flags
#define ZEROBITS		((unsigned int) 0)
#define BIT(n) 			((unsigned int) (1 << n))

#define ACCESS_NONE						ZEROBITS
#define ACCESS_READ_SATISFIED 			BIT( 0) 	// Read satisfiability
#define ACCESS_WRITE_SATISFIED 			BIT( 1) 	// Write satisfiability
#define ACCESS_UNREGISTERED	 			BIT( 2) 	// Unregistered
#define ACCESS_CHILDS_FINISHED			BIT( 3) 	// All the childs and its childs have finished
#define ACCESS_EARLY_READ				BIT( 4) 	// Read accesses can be released early
#define ACCESS_HASNEXT					BIT( 5)		// Has a ->_successor access
#define ACCESS_HASCHILD					BIT( 6)		// Has a ->_child access
#define ACCESS_NEXT_WRITE_SATISFIED		BIT( 7)		// Write satisfiability propagated to the next
#define ACCESS_NEXT_READ_SATISFIED		BIT( 8)		// Read satisfiability propagated to next
#define ACCESS_CHILD_WRITE_SATISFIED	BIT( 9) 	// Write satisfiability propagated to the child
#define ACCESS_CHILD_READ_SATISFIED		BIT(10) 	// Read satisfiability propagated to the read
#define ACCESS_PARENT_DONE				BIT(11)		// Parent has finished
#define ACCESS_NEXTISPARENT				BIT(12)		// Next = parent access
#define ACCESS_REDUCTION_COMBINED		BIT(13) 	// Combination checked
#define ACCESS_IS_WEAK					BIT(14) 	// Weak

typedef unsigned int access_flags_t;

enum PropagationDestination {
	NEXT,
	CHILD,
	PARENT,
	NONE
};

struct DataAccessMessage {
	access_flags_t flagsForNext; 			// Message to the next automata
	access_flags_t flagsAfterPropagation; 	// Flags to set again here
	DataAccess *from;						// Origin access
	DataAccess *to;							// Destination access
	bool schedule;							// Schedule this access
	bool combine;							// Combine reduction

	DataAccessMessage() :
		flagsForNext(ACCESS_NONE),
		flagsAfterPropagation(ACCESS_NONE),
		from(nullptr),
		to(nullptr),
		schedule(false),
		combine(false)
	{
	}
};

typedef std::stack<DataAccessMessage> mailbox_t;

#endif // DATA_ACCESS_FLAGS_HPP