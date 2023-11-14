#	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
#
#	Copyright (C) 2020-2023 Barcelona Supercomputing Center (BSC)

AC_DEFUN([AC_CHECK_JEMALLOC],
	[
		AC_ARG_WITH(
			[jemalloc],
			[AS_HELP_STRING([--with-jemalloc=prefix], [specify the installation prefix of jemalloc or "embedded" @<:@default=embedded@:>@])],
			[ ac_use_jemalloc_prefix=$withval ],
			[ ac_use_jemalloc_prefix="embedded" ]
		)

		jemalloc_LIBS=""
		jemalloc_build_LIBS=""
		jemalloc_CPPFLAGS=""

		ac_use_jemalloc=no
		ac_use_embedded_jemalloc=no

		AC_MSG_CHECKING([the jemalloc installation prefix])
		if test x"${ac_use_jemalloc_prefix}" = x"" ; then
			AC_MSG_RESULT([invalid prefix])
			AC_MSG_ERROR([jemalloc prefix specified but empty])
		elif test x"${ac_use_jemalloc_prefix}" = x"no" ; then
			AC_MSG_RESULT([not provided])
			AC_MSG_WARN([jemalloc not provided, expect performance penalties on memory allocations])
		elif test x"${ac_use_jemalloc_prefix}" = x"embedded" ; then
			AC_MSG_RESULT([using embedded jemalloc with nanos6 symbol prefix])

			AX_SUBDIRS_CONFIGURE([jemalloc_embedded_subdir],
				[[CFLAGS=-fPIC],
				[CXXFLAGS=-fPIC],
				[--prefix=${prefix}/jemalloc_embedded_install_subdir],
				[--with-install-suffix=-nanos6],
				[--with-jemalloc-prefix=nanos6_je_],
				[--enable-stats]])

			jemalloc_LIBS='-L$(top_builddir)/jemalloc_embedded_subdir/lib -Wl,--enable-new-dtags -Wl,-rpath=$(prefix)/jemalloc_embedded_install_subdir/lib -ljemalloc-nanos6'
			jemalloc_build_LIBS='-L$(top_builddir)/jemalloc_embedded_subdir/lib -Wl,--enable-new-dtags -Wl,-rpath=$(top_builddir)/jemalloc_embedded_subdir/lib -ljemalloc-nanos6'
			jemalloc_CPPFLAGS='-I$(top_builddir)/jemalloc_embedded_subdir/include'

			ac_use_jemalloc=yes
			ac_use_embedded_jemalloc=yes
		else
			AC_MSG_RESULT([${ac_use_jemalloc_prefix}])

			if test x"${ac_use_jemalloc_prefix}" != x"yes" ; then
				jemalloc_LIBS="-L${ac_use_jemalloc_prefix}/lib"
				jemalloc_CPPFLAGS="-I${ac_use_jemalloc_prefix}/include"
			fi

			ac_save_CPPFLAGS="${CPPFLAGS}"
			ac_save_LIBS="${LIBS}"

			CPPFLAGS="${CPPFLAGS} ${jemalloc_CPPFLAGS}"
			LIBS="${LIBS} ${jemalloc_LIBS}"

			AC_CHECK_HEADERS([jemalloc/jemalloc.h], [], [AC_MSG_ERROR([jemalloc jemalloc.h header file not found])])
			AC_CHECK_LIB([jemalloc],
				[nanos6_je_malloc],
				[
					jemalloc_LIBS="${jemalloc_LIBS} -ljemalloc"
					if test x"${ac_use_jemalloc_prefix}" != x"yes" ; then
						jemalloc_LIBS="${jemalloc_LIBS} -Wl,--enable-new-dtags -Wl,-rpath=${ac_use_jemalloc_prefix}/lib"
					fi
				],
				[AC_MSG_ERROR([jemalloc cannot be found])],
				[${ac_save_LIBS}]
			)

			CPPFLAGS="${ac_save_CPPFLAGS}"
			LIBS="${ac_save_LIBS}"

			jemalloc_build_LIBS="${jemalloc_LIBS}"
			ac_use_jemalloc=yes
		fi

		AM_CONDITIONAL(HAVE_JEMALLOC, test x"${ac_use_jemalloc}" = x"yes")
		AM_CONDITIONAL(USE_JEMALLOC_EMBEDDED, test x"${ac_use_embedded_jemalloc}" = x"yes")

		AC_SUBST([JEMALLOC_EMBEDDED_SUBDIR], [jemalloc_embedded_subdir])

		if test x"${ac_use_jemalloc}" = x"yes" ; then
			AC_DEFINE(HAVE_JEMALLOC, [1], [jemalloc API is available])
		fi
		if test x"${ac_use_embedded_jemalloc}" = x"yes" ; then
			AC_DEFINE(USE_JEMALLOC_EMBEDDED, [1], [jemalloc API is available])
		fi

		AC_SUBST([jemalloc_LIBS])
		AC_SUBST([jemalloc_build_LIBS])
		AC_SUBST([jemalloc_CPPFLAGS])
	]
)
