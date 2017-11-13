/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef NANOS6_REDUCTIONS_H
#define NANOS6_REDUCTIONS_H

#pragma GCC visibility push(default)


#ifdef __cplusplus
extern "C" {
#endif

enum ReductionOperation {
	RED_OP_ADDITION = 0,
	RED_OP_PRODUCT = 1,
	RED_OP_BITWISE_AND = 2,
	RED_OP_BITWISE_OR = 3,
	RED_OP_BITWISE_XOR = 4,
	RED_OP_LOGICAL_AND = 5,
	RED_OP_LOGICAL_OR = 6,
	RED_OP_LOGICAL_XOR = 7,
	RED_OP_LOGICAL_NXOR = 8,
	RED_OP_MAXIMUM = 9,
	RED_OP_MINIMUM = 10
};

enum ReductionType {
	RED_TYPE_CHAR = 1000,
	RED_TYPE_SIGNED_CHAR = 2000,
	RED_TYPE_UNSIGNED_CHAR = 3000,
	RED_TYPE_SHORT = 4000,
	RED_TYPE_UNSIGNED_SHORT = 5000,
	RED_TYPE_INT = 6000,
	RED_TYPE_UNSIGNED_INT = 7000,
	RED_TYPE_LONG = 8000,
	RED_TYPE_UNSIGNED_LONG = 9000,
	RED_TYPE_LONG_LONG = 10000,
	RED_TYPE_UNSIGNED_LONG_LONG = 11000,
	RED_TYPE_FLOAT = 12000,
	RED_TYPE_DOUBLE = 13000,
	RED_TYPE_LONG_DOUBLE = 14000,
	RED_TYPE_COMPLEX_FLOAT = 15000,
	RED_TYPE_COMPLEX_DOUBLE = 16000,
	RED_TYPE_COMPLEX_LONG_DOUBLE = 17000,
	RED_TYPE_BOOLEA = 18000
};


//! \brief Retrieve the original address of a reduction given the (possibly local)
//! reduction address set by the parent task on creation time
//!
//! \param[in] address the (possibly local) reduction address set by the parent task
//! \return the original address of the reduction
void *nanos_get_original_reduction_address(const void *address);


#ifdef __cplusplus
}
#endif

#pragma GCC visibility pop


#endif /* NANOS6_REDUCTIONS_H */
