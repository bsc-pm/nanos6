#	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
#	
#	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)

AC_DEFUN([AC_CHECK_MEMKIND],
	[
		AC_ARG_WITH(
			[memkind],
			[AS_HELP_STRING([--with-memkind=prefix], [specify the installation prefix of memkind])],
			[ ac_use_memkind_prefix=$withval ],
			[ ac_use_memkind_prefix="" ]
		)
		
		if test x"${ac_use_memkind_prefix}" != x"" ; then
			AC_MSG_CHECKING([the memkind installation prefix])
			AC_MSG_RESULT([${ac_use_memkind_prefix}])
			memkind_LIBS="-L${ac_use_memkind_prefix}/lib -lmemkind"
			memkind_CPPFLAGS="-I$ac_use_memkind_prefix/include"
			ac_use_memkind=yes
		else
			PKG_CHECK_MODULES(
				[memkind],
				[memkind],
				[
					AC_MSG_CHECKING([the memkind installation prefix])
					AC_MSG_RESULT([retrieved from pkg-config])
					memkind_CPPFLAGS="${memkind_CFLAGS}"
					ac_use_memkind=yes
				], [
					AC_MSG_CHECKING([the memkind installation prefix])
					AC_MSG_RESULT([not available])
				]
			)
		fi
		
		if test x"${ac_use_memkind}" = x"" ; then
			ac_save_CPPFLAGS="${CPPFLAGS}"
			ac_save_LIBS="${LIBS}"
			
			CPPFLAGS="${CPPFLAGS} ${memkind_CPPFLAGS}"
			LIBS="${LIBS} ${memkind_LIBS}"
			
			AC_CHECK_HEADERS([memkind.h])
			AC_CHECK_LIB([memkind],
				[memkind_malloc],
				[
					memkind_LIBS="${memkind_LIBS} -lmemkind"
					ac_use_memkind=yes
				],
				[
					if test x"${ac_use_memkind_prefix}" != x"" ; then
						AC_MSG_ERROR([memkind cannot be found.])
					else
						AC_MSG_WARN([memkind cannot be found.])
					fi
					ac_use_memkind=no
				]
			)
			
			CPPFLAGS="${ac_save_CPPFLAGS}"
			LIBS="${ac_save_LIBS}"
		fi
		
		if test "x${ac_use_memkind}" = x"yes"; then
			AC_DEFINE(HAVE_MEMKIND, 1, [use memkind])
		fi
		
		AM_CONDITIONAL(HAVE_MEMKIND, test x"${ac_use_memkind}" = x"yes")
		
		AC_SUBST([memkind_LIBS])
		AC_SUBST([memkind_CPPFLAGS])
	]
)
