#	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
#
#	Copyright (C) 2022 Barcelona Supercomputing Center (BSC)

AC_DEFUN([AC_CHECK_OVNI],
	[
		AC_ARG_WITH(
			[ovni],
			[AS_HELP_STRING([--with-ovni=prefix], [specify the installation prefix of the OVNI instrumentation library])],
			[ ac_cv_use_ovni_prefix=$withval ],
			[ ac_cv_use_ovni_prefix="" ]
		)

		if test x"${ac_cv_use_ovni_prefix}" = x"no"; then
			AC_MSG_CHECKING([the OVNI installation prefix])
			AC_MSG_RESULT([${ac_cv_use_ovni_prefix}])
			ac_use_ovni=no
		elif test x"${ac_cv_use_ovni_prefix}" != x"" ; then
			AC_MSG_CHECKING([the OVNI installation prefix])
			AC_MSG_RESULT([${ac_cv_use_ovni_prefix}])
			ovni_LIBS="-L${ac_cv_use_ovni_prefix}/lib -lovni -Wl,-rpath,${ac_cv_use_ovni_prefix}/lib"
			ovni_CPPFLAGS="-I$ac_cv_use_ovni_prefix/include"
			ac_use_ovni=yes
		fi

		if test x"${ac_use_ovni}" = x"yes" ; then
			ac_save_CPPFLAGS="${CPPFLAGS}"
			ac_save_LIBS="${LIBS}"

			CPPFLAGS="${CPPFLAGS} ${ovni_CPPFLAGS}"
			LIBS="${LIBS} ${ovni_LIBS}"

			AC_CHECK_HEADERS([ovni.h])
			AC_CHECK_LIB([ovni],
				[ovni_proc_init],
				[
					ovni_LIBS="${ovni_LIBS}"
					ac_use_ovni=yes
				],
				[
					if test x"${ac_cv_use_ovni_prefix}" != x"" ; then
						AC_MSG_ERROR([OVNI cannot be found.])
					else
						AC_MSG_WARN([OVNI cannot be found.])
					fi
					ac_use_ovni=no
				]
			)

			CPPFLAGS="${ac_save_CPPFLAGS}"
			LIBS="${ac_save_LIBS}"
		fi

		AM_CONDITIONAL(HAVE_OVNI, test x"${ac_use_ovni}" = x"yes")

		AC_SUBST([ovni_LIBS])
		AC_SUBST([ovni_CPPFLAGS])
	]
)
