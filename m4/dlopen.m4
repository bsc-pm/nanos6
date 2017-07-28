#	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
#	
#	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)

AC_DEFUN([AC_CHECK_DLOPEN],
	[
		ac_saved_LIBS="${LIBS}"
		LIBS=""
		AC_SEARCH_LIBS(
			[dlopen], [dl dld],
			[
				DLOPEN_LIBS="${LIBS}"
			],
			[
				AC_MSG_ERROR([unable to find the dlopen function])
			],
			[ ${ac_saved_LIBS} ]
		)
		AC_CHECK_FUNCS([dlinfo])
		LIBS="${ac_saved_LIBS}"
		
		AC_SUBST([DLOPEN_LIBS])
	]
)

