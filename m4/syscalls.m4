#	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
#	
#	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)

AC_DEFUN([AC_CHECK_MADV_FREE],
	[
		AC_LANG_PUSH(C)
		
		AC_MSG_CHECKING([if madvise supports the MADV_FREE flag])
		AC_COMPILE_IFELSE(
			[ AC_LANG_PROGRAM(
[[
#include <sys/mman.h>
]], [[
	int rc = madvise(0, 0, MADV_FREE);
	
	return 0;
]]
				) ],
			[  ac_have_madv_free=yes; AC_DEFINE([HAVE_MADV_FREE], 1, [madvise supports the MADV_FREE flag]) ],
			[ ac_have_madv_free=no ]
		)
		AC_MSG_RESULT([${ac_have_madv_free}])
		
		AC_LANG_POP(C)
	]
)


