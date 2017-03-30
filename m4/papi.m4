AC_DEFUN([AC_CHECK_PAPI],
	[
		AC_ARG_WITH([papi],
			[  --with-papi=[prefix]   set the PAPI installation to be used],
			[ ac_cv_use_papi_prefix=$withval ],
			[ ac_cv_use_papi_prefix="" ]
		)
		
		if test x"${ac_cv_use_papi_prefix}" != x"" ; then
			AC_MSG_CHECKING([the PAPI installation prefix])
			AC_MSG_RESULT([${ac_cv_use_papi_prefix}])
			papi_LIBS="-L${ac_cv_use_papi_prefix}/lib"
			papi_CPPFLAGS="-I$ac_cv_use_papi_prefix/include"
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
		
		if test x"${ac_use_papi}" = x"" ; then
			ac_save_CPPFLAGS="${CPPFLAGS}"
			ac_save_LIBS="${LIBS}"
			
			CPPFLAGS="${CPPFLAGS} ${papi_CPPFLAGS}"
			LIBS="${LIBS} ${papi_LIBS}"
			
			AC_CHECK_HEADERS([papi.h])
			AC_CHECK_LIB([papi],
				[PAPI_library_init],
				[
					papi_LIBS="${papi_LIBS} -lpapi"
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
		fi
		
		AM_CONDITIONAL(HAVE_PAPI, test x"${ac_use_papi}" = x"yes")
		
		AC_SUBST([papi_LIBS])
		AC_SUBST([papi_CPPFLAGS])
	]
)
