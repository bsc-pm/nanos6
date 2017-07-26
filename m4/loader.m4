#	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
#	
#	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)

AC_DEFUN([AC_CHECK_MAIN_WRAPPER_TYPE],
	[
		AC_LANG_PUSH(C)
		
		AC_MSG_CHECKING([if target is PowerPC])
		AC_COMPILE_IFELSE(
			[ AC_LANG_SOURCE( [[
#ifndef __powerpc__
# error not power
#endif
]]
				) ],
			[ ac_target_is_powerpc=yes ],
			[ ac_target_is_powerpc=no ]
		)
		AC_MSG_RESULT([${ac_target_is_powerpc}])
		
		AC_MSG_CHECKING([if target is Linux])
		AC_COMPILE_IFELSE(
			[ AC_LANG_SOURCE( [[
#ifndef __linux__
# error not linux
#endif
]]
				) ],
			[ ac_target_is_linux=yes ],
			[ ac_target_is_linux=no ]
		)
		AC_MSG_RESULT([${ac_target_is_linux}])
		
		AC_MSG_CHECKING([if target is Android])
		AC_COMPILE_IFELSE(
			[ AC_LANG_SOURCE( [[
#ifndef __ANDROID__
# error not android
#endif
]]
				) ],
			[ ac_target_is_android=yes ],
			[ ac_target_is_android=no ]
		)
		AC_MSG_RESULT([${ac_target_is_android}])
		
		AC_LANG_POP(C)
		
		AM_CONDITIONAL([LINUX_POWERPC_GLIBC], [test "x${ac_target_is_linux}" = "xyes" && test "x${ac_target_is_powerpc}" = "xyes" && test "x${ac_target_is_android}" = "xno"])
		AM_CONDITIONAL([LINUX_GLIBC], [test "x${ac_target_is_linux}" = "xyes" && test "x${ac_target_is_powerpc}" = "xno" && test "x${ac_target_is_android}" = "xno"])
		AM_CONDITIONAL([ANDROID], [test "x${ac_target_is_android}" = "xyes"])
	]
)


AC_DEFUN([AC_CHECK_SYMBOL_RESOLUTION_STRATEGY],
	[
		AC_MSG_CHECKING([which symbol resolution strategy to use])
		AC_ARG_WITH(
			[symbol-resolution],
			[AS_HELP_STRING([--with-symbol-resolution=ifunc|indirect], [specify the strategy to resolve the runtime symbols @<:@default=check@:>@])],
			[ac_cv_use_symbol_resolution="${withval}"],
			[ac_cv_use_symbol_resolution="check"]
		)
		
		if test "x${ac_cv_use_symbol_resolution}" = "xcheck" ; then
			AC_LANG_PUSH(C)
			AC_RUN_IFELSE(
				[AC_LANG_PROGRAM(
[[
#include <stdlib.h>

void (*_indirect_exit_resolver(void)) (int) {
	return exit;
}

void indirect_exit(int) __attribute__ (( ifunc("_indirect_exit_resolver") ));
]], [[
	indirect_exit(0);
	
	return 1;
]]
				)],
				[ac_cv_use_symbol_resolution=ifunc],
				[ac_cv_use_symbol_resolution=indirect],
				[AC_MSG_ERROR([cross-compilation detected. Please specify a symbol resolution method with --with-symbol-resolution=ifunc|indirect])]
			)
			AC_LANG_POP(C)
		fi
		
		case "x${ac_cv_use_symbol_resolution}" in
			"xifunc")
				AC_MSG_RESULT([${ac_cv_use_symbol_resolution}])
				;;
			"xindirect")
				AC_MSG_RESULT([${ac_cv_use_symbol_resolution}])
				;;
			*)
				AC_MSG_ERROR([Unknown loading strategy ${ac_cv_use_symbol_resolution}])
				;;
		esac
		
		AM_CONDITIONAL([RESOLVE_SYMBOLS_USING_IFUNC], [test "x${ac_cv_use_symbol_resolution}" = "xifunc"])
		AM_CONDITIONAL([RESOLVE_SYMBOLS_USING_INDIRECTION], [test "x${ac_cv_use_symbol_resolution}" = "xindirect"])
		
	]
)


