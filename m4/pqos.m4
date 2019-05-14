#	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
#	
#	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)

AC_DEFUN([AC_CHECK_PQOS],
	[
		AC_ARG_WITH(
			[pqos],
			[AS_HELP_STRING([--with-pqos=prefix], [specify the installation prefix of PQOS])],
			[ ac_cv_use_pqos_prefix=$withval ],
			[ ac_cv_use_pqos_prefix="" ]
		)
		
		if test x"${ac_cv_use_pqos_prefix}" != x"" ; then
			AC_MSG_CHECKING([the PQOS installation prefix])
			AC_MSG_RESULT([${ac_cv_use_pqos_prefix}])
			pqos_LIBS="-L${ac_cv_use_pqos_prefix}/lib -lpqos"
			pqos_CPPFLAGS="-I$ac_cv_use_pqos_prefix/include"
			ac_use_pqos=yes
		else
			PKG_CHECK_MODULES(
				[pqos],
				[pqos],
				[
					AC_MSG_CHECKING([the PQOS installation prefix])
					AC_MSG_RESULT([retrieved from pkg-config])
					pqos_CPPFLAGS="${pqos_CFLAGS}"
					ac_use_pqos=yes
				], [
					AC_MSG_CHECKING([the PQOS installation prefix])
					AC_MSG_RESULT([not available])
				]
			)
		fi
		
		if test x"${ac_use_pqos}" = x"" ; then
			ac_save_CPPFLAGS="${CPPFLAGS}"
			ac_save_LIBS="${LIBS}"
			
			CPPFLAGS="${CPPFLAGS} ${pqos_CPPFLAGS}"
			LIBS="${LIBS} ${pqos_LIBS}"
			
			AC_CHECK_HEADERS([pqos.h])
			AC_CHECK_LIB([pqos],
				[pqos_init],
				[
					pqos_LIBS="${pqos_LIBS} -lpqos"
					ac_use_pqos=yes
				],
				[
					if test x"${ac_cv_use_pqos_prefix}" != x"" ; then
						AC_MSG_ERROR([PQOS cannot be found.])
					else
						AC_MSG_WARN([PQOS cannot be found.])
					fi
					ac_use_pqos=no
				]
			)
			
			CPPFLAGS="${ac_save_CPPFLAGS}"
			LIBS="${ac_save_LIBS}"
		fi
		
		AM_CONDITIONAL(HAVE_PQOS, test x"${ac_use_pqos}" = x"yes")
		
		AC_SUBST([pqos_LIBS])
		AC_SUBST([pqos_CPPFLAGS])
	]
)
