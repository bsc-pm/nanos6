/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include <boost/assert.hpp>

#include "FatalErrorHandler.hpp"


#if !defined(NDEBUG)
namespace boost {
	void assertion_failed_msg(char const * expr, char const * msg, char const * function, char const * file, long line)
	{
		FatalErrorHandler::failIf<>(1, file, ":", line, " ", function, " Boost assertion failure: ", msg, " when evaluating ", expr);
	}
	
	void assertion_failed(char const * expr, char const * function, char const * file, long line)
	{
		FatalErrorHandler::failIf<>(1, file, ":", line, " ", function, " Boost assertion failure when evaluating ", expr);
	}
	
}
#endif

