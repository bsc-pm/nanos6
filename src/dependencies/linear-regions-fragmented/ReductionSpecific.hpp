/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef REDUCTION_SPECIFIC_HPP
#define REDUCTION_SPECIFIC_HPP


#include <limits.h>


typedef int reduction_type_and_operator_index_t;
typedef int reduction_index_t;


enum specialReductionTypeAndOperatorIndexes_t {
	no_reduction_type_and_operator = INT_MAX
};

enum specialReductionIndexes_t {
	no_reduction_index = INT_MAX
};


#endif // REDUCTION_SPECIFIC_HPP
