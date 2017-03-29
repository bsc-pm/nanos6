AC_DEFUN([AC_CHECK_ARGOBOTS],
	[
		ARGOBOTS_LDFLAGS=
		AC_ARG_WITH([argobots],
			[  --with-argobots=[prefix]   set the argobots installation to be used],
			[
				ac_cv_use_argobots_prefix=$withval
				ARGOBOTS_LDFLAGS="-L$ac_cv_use_argobots_prefix/lib -L$ac_cv_use_argobots_prefix/lib64 $LIBS"
				CPPFLAGS="-I$ac_cv_use_argobots_prefix/include $CPPFLAGS"
			],
			[
				ac_cv_use_argobots_prefix=""
			]
		)

		saved_libs="$LIBS"
		LIBS="$ARGOBOTS_LDFLAGS"
		AC_CHECK_HEADERS([abt.h])
		AC_CHECK_LIB([abt],
			[ ABT_init ],
			[
				ARGOBOTS_LDFLAGS="$ARGOBOTS_LDFLAGS -labt"
			],
			[ AC_MSG_ERROR([argobots cannot be found. Please use the --with-argobots parameter if necessary.]) ]
		)
		LIBS="$saved_libs"
		AC_SUBST(ARGOBOTS_LDFLAGS)
	]
)
