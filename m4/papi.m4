#	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
#
#	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)

AC_DEFUN([AC_CHECK_PAPI],
	[
		AC_ARG_WITH(
			[papi],
			[AS_HELP_STRING([--with-papi=prefix], [specify the installation prefix of PAPI])],
			[ ac_cv_use_papi_prefix=$withval ],
			[ ac_cv_use_papi_prefix="" ]
		)

		if test x"${ac_cv_use_papi_prefix}" != x"" ; then
			AC_MSG_CHECKING([the PAPI installation prefix])
			AC_MSG_RESULT([${ac_cv_use_papi_prefix}])
			papi_LIBS="-L${ac_cv_use_papi_prefix}/lib -lpapi -Wl,-rpath,${ac_cv_use_papi_prefix}/lib"
			papi_CPPFLAGS="-I$ac_cv_use_papi_prefix/include"
			ac_use_papi=yes
		else
			PKG_CHECK_MODULES(
				[papi],
				[papi],
				[
					AC_MSG_CHECKING([the PAPI installation prefix])
					AC_MSG_RESULT([retrieved from pkg-config])
					papi_CPPFLAGS="${papi_CFLAGS}"
					ac_use_papi=yes
				], [
					AC_MSG_CHECKING([the PAPI installation prefix])
					AC_MSG_RESULT([not available])
				]
			)
		fi

		if test x"${ac_use_papi}" != x"" ; then
			ac_save_CPPFLAGS="${CPPFLAGS}"
			ac_save_LIBS="${LIBS}"

			CPPFLAGS="${CPPFLAGS} ${papi_CPPFLAGS}"
			LIBS="${LIBS} ${papi_LIBS}"

			AC_CHECK_HEADERS([papi.h])
			AC_CHECK_LIB([papi],
				[PAPI_library_init],
				[
					papi_LIBS="${papi_LIBS}"
					ac_use_papi=yes
				],
				[
					if test x"${ac_cv_use_papi_prefix}" != x"" ; then
						AC_MSG_ERROR([PAPI cannot be found.])
					else
						AC_MSG_WARN([PAPI cannot be found.])
					fi
					ac_use_papi=no
				]
			)

			CPPFLAGS="${ac_save_CPPFLAGS}"
			LIBS="${ac_save_LIBS}"
		fi

		if test x"${ac_use_papi}" != x"" ; then
			if test x"${ac_cv_use_papi_prefix}" != x"" ; then
				papiBinary=${ac_cv_use_papi_prefix}/bin/papi_version
			else
				papiBinary=papi_version
			fi
			papiVersion=`$papiBinary | sed 's/[[^0-9.]]*\([[0-9.]]*\).*/\1/'`

			AX_COMPARE_VERSION(
				[[${papiVersion}]],
				[[ge]],
				[[5.6.0]],
				[[ac_papi_version_correct=yes]],
				[[ac_papi_version_correct=no]]
			)

			if test x"${ac_papi_version_correct}" != x"yes" ; then
				AC_MSG_ERROR([PAPI version must be >= 5.6.0.])
				ac_use_papi=no
			else
				AC_MSG_CHECKING([if the PAPI version >= 5.6.0.])
				AC_MSG_RESULT([${ac_papi_version_correct}])
			fi
		fi

		AM_CONDITIONAL(HAVE_PAPI, test x"${ac_use_papi}" = x"yes")

		AC_SUBST([papi_LIBS])
		AC_SUBST([papi_CPPFLAGS])
	]
)
