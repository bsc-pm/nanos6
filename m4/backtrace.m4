#	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
#	
#	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)

AC_DEFUN([AC_CHECK_BACKTRACE],
	[
		AC_ARG_WITH([libunwind],
			[  --with-libunwind=[prefix]   set the libunwind installation to be used],
			[
				ac_cv_use_libunwind_prefix=$withval
				if test -d "$ac_cv_use_libunwind_prefix/lib" ; then
					LIBS="-L$ac_cv_use_libunwind_prefix/lib $LIBS"
				elif test -d "$ac_cv_use_libunwind_prefix/lib64" ; then
					LIBS="-L$ac_cv_use_libunwind_prefix/lib64 $LIBS"
				elif test -d "$ac_cv_use_libunwind_prefix/lib32" ; then
					LIBS="-L$ac_cv_use_libunwind_prefix/lib32 $LIBS"
				else
					AC_MSG_WARN([cannot find libunwind library directory])
				fi
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
				if test x"${ac_cv_func_backtrace}" ; then
					AC_MSG_WARN([libunwind cannot be found. Will use backtrace from libc instead.])
				else
					AC_MSG_WARN([libunwind cannot be found. Unwind information will not be available.])
				fi
				AC_DEFINE([HAVE_LIBUNWIND], 0, [use libunwind to generate backtraces])
			]
		)
		
		AC_SUBST([BACKTRACE_LIBS])
		
		AM_CONDITIONAL([HAVE_LIBUNWIND], [test x"${ac_cv_header_libunwind_h}" = x"yes" -a x"${ac_cv_lib_unwind_backtrace}" = x"yes"])
		AM_CONDITIONAL([HAVE_BACKTRACE], [test x"${ac_cv_header_execinfo_h}" = x"yes" -a x"${ac_cv_func_backtrace}" = x"yes"])
	]
)
