#	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
#
#	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)

AC_DEFUN([AC_CHECK_JEMALLOC],
	[
		AC_ARG_WITH(
			[jemalloc],
			[AS_HELP_STRING([--with-jemalloc=prefix], [specify the installation prefix of jemalloc])],
			[ ac_cv_use_jemalloc_prefix=$withval ],
			[ ac_cv_use_jemalloc_prefix="" ]
		)

		if test x"${ac_cv_use_jemalloc_prefix}" != x"" ; then
			AC_MSG_CHECKING([the jemalloc installation prefix])
			AC_MSG_RESULT([${ac_cv_use_jemalloc_prefix}])
			jemalloc_LIBS="-L${ac_cv_use_jemalloc_prefix}/lib -ljemalloc"
			jemalloc_CPPFLAGS="-I$ac_cv_use_jemalloc_prefix/include"
			ac_use_jemalloc=yes
		else
			PKG_CHECK_MODULES(
				[jemalloc],
				[jemalloc],
				[
					AC_MSG_CHECKING([the jemalloc installation prefix])
					AC_MSG_RESULT([retrieved from pkg-config])
					jemalloc_CPPFLAGS="${jemalloc_CFLAGS}"
					ac_use_jemalloc=yes
				], [
					AC_MSG_CHECKING([the jemalloc installation prefix])
					AC_MSG_RESULT([not available])
				]
			)
		fi

		if test x"${ac_use_jemalloc}" != x"" ; then
			ac_save_CPPFLAGS="${CPPFLAGS}"
			ac_save_LIBS="${LIBS}"

			CPPFLAGS="${CPPFLAGS} ${jemalloc_CPPFLAGS}"
			LIBS="${LIBS} ${jemalloc_LIBS}"

			AC_CHECK_HEADERS([jemalloc/jemalloc.h])
			AC_CHECK_LIB([jemalloc],
				[nanos6_je_malloc],
				[
					jemalloc_LIBS="${jemalloc_LIBS}"
					ac_use_jemalloc=yes
				],
				[
					if test x"${ac_cv_use_jemalloc_prefix}" != x"" ; then
						AC_MSG_ERROR([jemalloc cannot be found.])
					else
						AC_MSG_WARN([jemalloc cannot be found.])
					fi
					ac_use_jemalloc=no
				]
			)

			CPPFLAGS="${ac_save_CPPFLAGS}"
			LIBS="${ac_save_LIBS}"
		fi

		AM_CONDITIONAL(HAVE_JEMALLOC, test x"${ac_use_jemalloc}" = x"yes")

		if test x"${ac_use_jemalloc}" = x"yes" ; then
			AC_DEFINE(HAVE_JEMALLOC, [1], [jemalloc API is available])
		fi

		AC_SUBST([jemalloc_LIBS])
		AC_SUBST([jemalloc_CPPFLAGS])
	]
)
