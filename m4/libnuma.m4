#	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
#	
#	Copyright (C) 2015-2018 Barcelona Supercomputing Center (BSC)

AC_DEFUN([AC_CHECK_LIBNUMA],
	[
		AC_ARG_WITH(
			[libnuma],
			[AS_HELP_STRING([--with-libnuma=prefix], [specify the installation prefix of the numactl library])],
			[ ac_use_libnuma_prefix="${withval}" ],
			[ ac_use_libnuma_prefix="" ]
		)
		
		if test x"${ac_use_libnuma_prefix}" != x"" ; then
			AC_MSG_CHECKING([the libnuma installation prefix])
			AC_MSG_RESULT([${ac_use_libnuma_prefix}])
			libnuma_LIBS="-L${ac_use_libnuma_prefix}/lib"
			libnuma_CPPFLAGS="-I${ac_use_libnuma_prefix}/include"
		fi
		
		ac_save_CPPFLAGS="${CPPFLAGS}"
		ac_save_LIBS="${LIBS}"
		
		CPPFLAGS="${CPPFLAGS} ${libnuma_CPPFLAGS}"
		LIBS="${LIBS} ${libnuma_LIBS}"
		
		AC_CHECK_HEADERS([numa.h], [], [AC_MSG_ERROR([libnuma header files cannot be found])])
		AC_CHECK_LIB([numa],
			[numa_alloc_interleaved],
			[libnuma_LIBS="${libnuma_LIBS} -lnuma"],
			[AC_MSG_ERROR([libnuma cannot be found])],
			[${ac_save_LIBS}]
		)
		
		CPPFLAGS="${ac_save_CPPFLAGS}"
		LIBS="${ac_save_LIBS}"
		
		AC_SUBST([libnuma_LIBS])
		AC_SUBST([libnuma_CPPFLAGS])
	]
)

