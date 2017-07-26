#	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
#	
#	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)

AC_DEFUN([SSS_CHECK_SANE_CHRONO],
	[
		AC_MSG_CHECKING([if the C++ implementation of chrono is usable])
		AC_LANG_PUSH(C++)
		AC_TRY_COMPILE(
			[ #include <chrono> ],
			[
			    std::chrono::system_clock::time_point __s_entry = std::chrono::system_clock::now();
			    std::chrono::nanoseconds __delta;
			    std::chrono::system_clock::time_point __s_atime = __s_entry + __delta;
			],
			[ AC_MSG_RESULT([yes]) ],
			[
				AC_MSG_RESULT([no])
				AC_MSG_ERROR([$CXX uses an implementation of C++11 chrono that has serious bugs, probably the GCC 4.7 implementation])
			]
		)
		AC_LANG_POP(C++)
	]
)


