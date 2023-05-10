#	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
#
#	Copyright (C) 2021-2022 Barcelona Supercomputing Center (BSC)

AC_DEFUN([AC_CHECK_HWLOC],
	[
		AC_ARG_WITH(
			[hwloc],
			[AS_HELP_STRING([--with-hwloc@<:@=DIR@:>@], [specify the installation prefix of the hwloc library])],
			[ ac_use_hwloc_prefix="${withval}" ],
			[ ac_use_hwloc_prefix="no" ]
		)

		hwloc_LIBS=""
		hwloc_LIBADD=""
		hwloc_CPPFLAGS=""
		ac_use_hwloc=no

		AC_MSG_CHECKING([the hwloc installation prefix])
		if test x"${ac_use_hwloc_prefix}" = x"no" ; then
			ac_internal_hwloc_prefix='nanos6_'
			AC_MSG_RESULT([using internal hwloc with ${ac_internal_hwloc_prefix} prefix])

			AC_DEFINE([USE_INTERNAL_HWLOC], [1], [Use internal hwloc])
			HWLOC_SET_SYMBOL_PREFIX(${ac_internal_hwloc_prefix})
			HWLOC_SETUP_CORE([hwloc-2.3.0], [happy=yes], [happy=no])
				AS_IF([test "$happy" = "no"],
				[AC_MSG_ERROR([Cannot continue])])

			hwloc_CPPFLAGS=${HWLOC_EMBEDDED_CPPFLAGS}
			hwloc_LIBS=${HWLOC_EMBEDDED_LIBS}
			hwloc_LIBADD=${HWLOC_EMBEDDED_LDADD}

		elif test x"${ac_use_hwloc_prefix}" = x"" ; then
			AC_MSG_RESULT([invalid prefix])
			AC_MSG_ERROR([hwloc prefix specified but empty])
		else
			AC_MSG_RESULT([${ac_use_hwloc_prefix}])

			if test x"${ac_use_hwloc_prefix}" != x"yes" ; then
				hwloc_LIBS="-L${ac_use_hwloc_prefix}/lib"
				hwloc_CPPFLAGS="-I${ac_use_hwloc_prefix}/include"
			fi

			ac_save_CPPFLAGS="${CPPFLAGS}"
			ac_save_LIBS="${LIBS}"

			CPPFLAGS="${CPPFLAGS} ${hwloc_CPPFLAGS}"
			LIBS="${LIBS} ${hwloc_LIBS}"

			AC_CHECK_HEADERS([hwloc.h], [], [AC_MSG_ERROR([hwloc hwloc.h header file not found])])
			AC_CHECK_LIB([hwloc],
				[hwloc_get_api_version],
				[hwloc_LIBS="${hwloc_LIBS} -lhwloc -Wl,--enable-new-dtags -Wl,-rpath=${ac_use_hwloc_prefix}/lib"],
				[AC_MSG_ERROR([hwloc cannot be found])],
				[${ac_save_LIBS}]
			)

			ac_use_hwloc=yes

			CPPFLAGS="${ac_save_CPPFLAGS}"
			LIBS="${ac_save_LIBS}"
		fi
		HWLOC_DO_AM_CONDITIONALS

		AM_CONDITIONAL(HAVE_HWLOC, test x"${ac_use_hwloc}" = x"yes")
		AC_SUBST([hwloc_LIBS])
		AC_SUBST([hwloc_LIBADD])
		AC_SUBST([hwloc_CPPFLAGS])
	]
)
