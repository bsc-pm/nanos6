/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef NULL_WISDOM_MANAGER_HPP
#define NULL_WISDOM_MANAGER_HPP

class WisdomManager {

public:
	
	static inline void initialize()
	{
	}
	
	static inline void shutdown()
	{
	}
	
	static inline bool isEnabled()
	{
		return false;
	}
	
};

#endif // NULL_WISDOM_MANAGER_HPP
