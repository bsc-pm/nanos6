#	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
#
#	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)

AC_DEFUN([AC_CHECK_DLB],
	[
		AC_ARG_WITH(
			[dlb],
			[AS_HELP_STRING([--with-dlb=prefix], [specify the installation prefix of DLB])],
			[ ac_cv_use_dlb_prefix=$withval ],
			[ ac_cv_use_dlb_prefix="" ]
		)

		if test x"${ac_cv_use_dlb_prefix}" != x"" ; then
			AC_MSG_CHECKING([the DLB installation prefix])
			AC_MSG_RESULT([${ac_cv_use_dlb_prefix}])
			dlb_LIBS="-L${ac_cv_use_dlb_prefix}/lib -ldlb"
			dlb_CPPFLAGS="-I$ac_cv_use_dlb_prefix/include"
			ac_use_dlb=yes
		else
			PKG_CHECK_MODULES(
				[dlb],
				[dlb],
				[
					AC_MSG_CHECKING([the DLB installation prefix])
					AC_MSG_RESULT([retrieved from pkg-config])
					dlb_CPPFLAGS="${dlb_CFLAGS}"
					ac_use_dlb=yes
				], [
					AC_MSG_CHECKING([the DLB installation prefix])
					AC_MSG_RESULT([not available])
				]
			)
		fi

		if test x"${ac_use_dlb}" != x"" ; then
			ac_save_CPPFLAGS="${CPPFLAGS}"
			ac_save_LIBS="${LIBS}"

			CPPFLAGS="${CPPFLAGS} ${dlb_CPPFLAGS}"
			LIBS="${LIBS} ${dlb_LIBS}"

			AC_CHECK_HEADERS([dlb.h])
			AC_CHECK_LIB([dlb],
				[DLB_Init],
				[
					dlb_LIBS="${dlb_LIBS}"
					ac_use_dlb=yes
				],
				[
					if test x"${ac_cv_use_dlb_prefix}" != x"" ; then
						AC_MSG_ERROR([DLB cannot be found.])
					else
						AC_MSG_WARN([DLB cannot be found.])
					fi
					ac_use_dlb=no
				]
			)

			CPPFLAGS="${ac_save_CPPFLAGS}"
			LIBS="${ac_save_LIBS}"
		fi

		AM_CONDITIONAL(HAVE_DLB, test x"${ac_use_dlb}" = x"yes")

		AC_SUBST([dlb_LIBS])
		AC_SUBST([dlb_CPPFLAGS])
	]
)
