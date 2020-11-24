/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef DATA_ACCESS_FLAGS_HPP
#define DATA_ACCESS_FLAGS_HPP

#include <cstdint>

#include "support/Containers.hpp"

struct DataAccess;

//! Definitions for the access flags
#define ZEROBITS           ((unsigned int) 0)
#define BIT(n)             ((unsigned int) (1 << n))

#define ACCESS_NONE                            ZEROBITS
#define ACCESS_READ_SATISFIED                  BIT( 0)     // Read satisfiability
#define ACCESS_WRITE_SATISFIED                 BIT( 1)     // Write satisfiability
#define ACCESS_CONCURRENT_SATISFIED            BIT( 2)     // *FUTURE* Concurrent satisfiability
#define ACCESS_COMMUTATIVE_SATISFIED           BIT( 3)     // *FUTURE* Commutative satisfiability
#define ACCESS_UNREGISTERED                    BIT( 4)     // Unregistered
#define ACCESS_CHILD_WRITE_DONE                BIT( 5)     // The child access has released writes
#define ACCESS_CHILD_READ_DONE                 BIT( 6)     // The child access has released reads
#define ACCESS_CHILD_CONCURRENT_DONE           BIT( 7)     // *FUTURE* The child access has released concurrents
#define ACCESS_CHILD_COMMUTATIVE_DONE          BIT( 8)     // *FUTURE* The child access has released commutatives
#define ACCESS_HASNEXT                         BIT( 9)     // Has a ->_successor access
#define ACCESS_HASCHILD                        BIT(10)     // Has a ->_child access
#define ACCESS_NEXT_WRITE_SATISFIED            BIT(11)     // Write satisfiability propagated to the next
#define ACCESS_NEXT_READ_SATISFIED             BIT(12)     // Read satisfiability propagated to next
#define ACCESS_NEXT_CONCURRENT_SATISFIED       BIT(13)     // *FUTURE* Concurrent satisfiability propagated to the next
#define ACCESS_NEXT_COMMUTATIVE_SATISFIED      BIT(14)     // *FUTURE* Commutative satisfiability propagated to the next
#define ACCESS_CHILD_WRITE_SATISFIED           BIT(15)     // Write satisfiability propagated to the child
#define ACCESS_CHILD_READ_SATISFIED            BIT(16)     // Read satisfiability propagated to the child
#define ACCESS_CHILD_CONCURRENT_SATISFIED      BIT(17)     // *FUTURE* Concurrent satisfiability propagated to the child
#define ACCESS_CHILD_COMMUTATIVE_SATISFIED     BIT(18)     // *FUTURE* Commutative satisfiability propagated to the child
#define ACCESS_PARENT_DONE                     BIT(19)     // Parent has finished
#define ACCESS_NEXTISPARENT                    BIT(20)     // Next = parent access
#define ACCESS_REDUCTION_COMBINED              BIT(21)     // Combination checked
#define ACCESS_IS_WEAK                         BIT(22)     // Is a weak access (non-atomic, just to save space)

typedef uint32_t access_flags_t;

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

typedef Container::stack<DataAccessMessage> mailbox_t;

#endif // DATA_ACCESS_FLAGS_HPP
