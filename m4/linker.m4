#	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
#	
#	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)

AC_DEFUN([CHECK_UNDEFINED_SYMBOL_VERIFICATION_FLAGS],
	[
		AC_MSG_CHECKING([if the linker accepts flags to fail on undefined symbols])
		AC_LANG_PUSH(C)
		ac_save_LDFLAGS="${LDFLAGS}"
		LDFLAGS="${LDFLAGS} -Wl,-z,defs"
		AC_LINK_IFELSE(
			[AC_LANG_PROGRAM([[]], [[]])],
			[
				AC_MSG_RESULT([yes])
				LDFLAGS="${ac_save_LDFLAGS}"
				LDFLAGS_NOUNDEFINED="-Wl,-z,defs"
			], [
				AC_MSG_RESULT([no])
				LDFLAGS="${ac_save_LDFLAGS}"
				LDFLAGS_NOUNDEFINED=""
			]
		)
		AC_LANG_POP(C)
		
		AC_SUBST(LDFLAGS_NOUNDEFINED)
	]
)
