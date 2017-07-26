/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef RUNTIME_INFO_ESSENTIALS_HPP
#define RUNTIME_INFO_ESSENTIALS_HPP


class RuntimeInfoEssentials {
public:
	static void initialize();
	static inline void shutdown()
	{
	}
};


#endif // RUNTIME_INFO_ESSENTIALS_HPP
