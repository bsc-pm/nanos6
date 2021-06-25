#	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
#
#	Copyright (C) 2021 Barcelona Supercomputing Center (BSC)

AC_DEFUN([AC_CHECK_BABELTRACE2], [
	AC_ARG_WITH(
		[babeltrace2],
		[AS_HELP_STRING(
			[--with-babeltrace2=prefix],
			[specify the installation prefix of babeltrace2 required by the fast CTF converter]
		)],
		[ ac_cv_use_babeltrace2_prefix=$withval ],
		[ ac_cv_use_babeltrace2_prefix="" ]
	)

	babeltrace2_LIBS=""
	babeltrace2_CPPFLAGS=""

	# Use babeltrace2 from a custom installation
	if test x"${ac_cv_use_babeltrace2_prefix}" = x"no"; then
		AC_MSG_CHECKING([the babeltrace2 installation prefix])
		AC_MSG_RESULT([${ac_cv_use_babeltrace2_prefix}])
		ac_use_babeltrace2=no
	elif test x"${ac_cv_use_babeltrace2_prefix}" != x"" ; then
		AC_MSG_CHECKING([the babeltrace2 installation prefix])
		AC_MSG_RESULT([${ac_cv_use_babeltrace2_prefix}])
		babeltrace2_LIBS="-L${ac_cv_use_babeltrace2_prefix}/lib"
		babeltrace2_CPPFLAGS="-I$ac_cv_use_babeltrace2_prefix/include"
		ac_use_babeltrace2=yes
	fi

	if test x"${ac_use_babeltrace2}" != x"" ; then
		# Link with babeltrace2
		babeltrace2_LIBS="${babeltrace2_LIBS} -lbabeltrace2"

		# Save flags to temporal variable
		ac_save_CPPFLAGS="${CPPFLAGS}"
		ac_save_LIBS="${LIBS}"

		# Add babeltrace flags
		CPPFLAGS="${CPPFLAGS} ${babeltrace2_CPPFLAGS}"
		LIBS="${LIBS} ${babeltrace2_LIBS}"

		# Check babeltrace2 headers and library
		AC_CHECK_HEADERS([babeltrace2/babeltrace.h])
		AC_CHECK_LIB([babeltrace2],
			[bt_graph_run],
			[
				babeltrace2_LIBS="${babeltrace2_LIBS}"
				ac_use_babeltrace=yes
			],
			[
				if test x"${ac_cv_use_babeltrace2_prefix}" != x"" ; then
					AC_MSG_ERROR([babeltrace2 cannot be found])
				else
					AC_MSG_WARN([babeltrace2 cannot be found])
				fi
				ac_use_babeltrace2=no
			]
		)

		# Restore original flags
		CPPFLAGS="${ac_save_CPPFLAGS}"
		LIBS="${ac_save_LIBS}"
	fi

	AM_CONDITIONAL(HAVE_BABELTRACE2, test x"${ac_use_babeltrace2}" = x"yes")

	if test x"${ac_use_babeltrace2}" = x"yes" ; then
		AC_DEFINE(HAVE_BABELTRACE2, [1], [babeltrace2 API is available])
	fi

	AC_SUBST([babeltrace2_LIBS])
	AC_SUBST([babeltrace2_CPPFLAGS])
])
