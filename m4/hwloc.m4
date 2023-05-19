#	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
#
#	Copyright (C) 2023 Barcelona Supercomputing Center (BSC)

AC_DEFUN([AC_CHECK_HWLOC],
	[
		AC_ARG_WITH(
			[hwloc],
			[AS_HELP_STRING(
				[--with-hwloc@<:@=DIR@:>@],
				[specify the installation prefix of the hwloc library. Values may be the prefix, "embedded", or "pkgconfig" @<:@default=pkgconfig@:>@]
			)],
			[ ac_use_hwloc_prefix="${withval}" ],
			[ ac_use_hwloc_prefix="pkgconfig" ]
		)

		hwloc_LIBS=""
		hwloc_LIBADD=""
		hwloc_CPPFLAGS=""
		hwloc_CFLAGS=""

		ac_use_embedded_hwloc=no

		AC_MSG_CHECKING([the hwloc installation prefix])
		if test x"${ac_use_hwloc_prefix}" = x"" ; then
			AC_MSG_RESULT([invalid prefix])
			AC_MSG_ERROR([hwloc prefix specified but empty])
		elif test x"${ac_use_hwloc_prefix}" = x"no" ; then
			AC_MSG_RESULT([not provided])
			AC_MSG_WARN([hwloc not provided, compilation may fail])
		elif test x"${ac_use_hwloc_prefix}" = x"pkgconfig" ; then
			AC_MSG_RESULT([using pkgconfig for discovery])
			PKG_CHECK_MODULES([hwloc], [hwloc])
		elif test x"${ac_use_hwloc_prefix}" = x"embedded" ; then
			ac_use_embedded_hwloc=yes
			ac_embedded_hwloc_symbol_prefix='nanos6_'
			AC_MSG_RESULT([using embedded hwloc with ${ac_embedded_hwloc_symbol_prefix} symbol prefix])

			HWLOC_SET_SYMBOL_PREFIX(${ac_embedded_hwloc_symbol_prefix})
			HWLOC_SETUP_CORE([hwloc_embedded_subdir], [happy=yes], [happy=no])
				AS_IF([test "$happy" = "no"],
				[AC_MSG_ERROR([cannot configure embedded hwloc])])

			hwloc_CPPFLAGS=${HWLOC_EMBEDDED_CPPFLAGS}
			hwloc_LIBS=${HWLOC_EMBEDDED_LIBS}
			hwloc_LIBADD=${HWLOC_EMBEDDED_LDADD}
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
				[
					hwloc_LIBS="${hwloc_LIBS} -lhwloc"
					if test x"${ac_use_hwloc_prefix}" != x"yes" ; then
						hwloc_LIBS="${hwloc_LIBS} -Wl,--enable-new-dtags -Wl,-rpath=${ac_use_hwloc_prefix}/lib"
					fi
				],
				[AC_MSG_ERROR([hwloc cannot be found])],
				[${ac_save_LIBS}]
			)

			CPPFLAGS="${ac_save_CPPFLAGS}"
			LIBS="${ac_save_LIBS}"
		fi
		HWLOC_DO_AM_CONDITIONALS

		AM_CONDITIONAL(USE_HWLOC_EMBEDDED, test x"${ac_use_embedded_hwloc}" = x"yes")

		AC_SUBST([HWLOC_EMBEDDED_SUBDIR], [hwloc_embedded_subdir])

		AC_SUBST([hwloc_LIBS])
		AC_SUBST([hwloc_LIBADD])
		AC_SUBST([hwloc_CPPFLAGS])
		AC_SUBST([hwloc_CFLAGS])
	]
)
