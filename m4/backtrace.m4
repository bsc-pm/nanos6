AC_DEFUN([AC_CHECK_BACKTRACE],
	[
		AC_ARG_WITH([libunwind],
			[  --with-libunwind=[prefix]   set the libunwind installation to be used],
			[
				ac_cv_use_libunwind_prefix=$withval
				LIBS="-L$ac_cv_use_libunwind_prefix/lib $LIBS"
				CPPFLAGS="-I$ac_cv_use_libunwind_prefix/include $CPPFLAGS"
			],
			[
				ac_cv_use_libunwind_prefix=""
			]
		)
		
		AC_CHECK_HEADERS([execinfo.h libunwind.h])
		AC_CHECK_FUNCS([backtrace])
		
		AC_CHECK_LIB([unwind],
			[backtrace],
			[
				BACKTRACE_LIBS="${BACKTRACE_LIBS} -lunwind"
				AC_DEFINE([HAVE_LIBUNWIND], 1, [use libunwind to generate backtraces])
			],
			[
				AC_MSG_WARN([libunwind cannot be found. Unwind information will not be available.])
				AC_DEFINE([HAVE_LIBUNWIND], 0, [use libunwind to generate backtraces])
			]
		)
		
		AC_SUBST([BACKTRACE_LIBS])
	]
)
