#include <boost/assert.hpp>

#include "FatalErrorHandler.hpp"


#if !defined(NDEBUG)
namespace boost {
	void assertion_failed_msg(char const * expr, char const * msg, char const * function, char const * file, long line)
	{
		FatalErrorHandler::handle<>(1, file, ":", line, " ", function, " Boost assertion failure: ", msg, " when evaluating ", expr);
	}
	
	void assertion_failed(char const * expr, char const * function, char const * file, long line)
	{
		FatalErrorHandler::handle<>(1, file, ":", line, " ", function, " Boost assertion failure when evaluating ", expr);
	}
	
}
#endif

