/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef DATA_ACCESS_TYPE_HPP
#define DATA_ACCESS_TYPE_HPP


typedef enum {
	NO_ACCESS_TYPE = 0,
	READ_ACCESS_TYPE,
	WRITE_ACCESS_TYPE,
	READWRITE_ACCESS_TYPE,
	CONCURRENT_ACCESS_TYPE,
	COMMUTATIVE_ACCESS_TYPE,
	REDUCTION_ACCESS_TYPE
} DataAccessType;


#endif // DATA_ACCESS_TYPE_HPP
