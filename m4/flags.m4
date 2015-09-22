
AC_DEFUN([AC_CHECK_COMPILER_FLAG],
	[
		
		ac_save_[]_AC_LANG_PREFIX[]FLAGS="$[]_AC_LANG_PREFIX[]FLAGS"
		AC_MSG_CHECKING([if $[]_AC_CC[] $[]_AC_LANG_PREFIX[]FLAGS supports the $1 flag])
		_AC_LANG_PREFIX[]FLAGS="$[]_AC_LANG_PREFIX[]FLAGS $1"
		AC_LINK_IFELSE(
			[AC_LANG_PROGRAM([[]], [[]])],
			[
				AC_MSG_RESULT([yes])
			], [
				AC_MSG_RESULT([no])
				_AC_LANG_PREFIX[]FLAGS="$ac_save_[]_AC_LANG_PREFIX[]FLAGS"
			]
		)
		
	]
)


AC_DEFUN([AC_CHECK_COMPILER_FLAGS],
	[
		for flag in $1 ; do
			AC_CHECK_COMPILER_FLAG([$flag])
		done
	]
)


AC_DEFUN([AC_CHECK_FIRST_COMPILER_FLAG],
	[
		ac_save2_[]_AC_LANG_PREFIX[]FLAGS="$[]_AC_LANG_PREFIX[]FLAGS"
		for flag in $1 ; do
			AC_CHECK_COMPILER_FLAG([$flag])
			if test x"$ac_save2_[]_AC_LANG_PREFIX[]FLAGS" != x"$[]_AC_LANG_PREFIX[]FLAGS" ; then
				break;
			fi
		done
	]
)


# This should be called before AC_PROG_CXX
AC_DEFUN([SSS_PREPARE_COMPILER_FLAGS],
	[
		AC_ARG_VAR(DEBUG_CXXFLAGS, [C++ compiler flags for debugging versions])
		
		user_CXXFLAGS="${CXXFLAGS}"
		# Do not let autoconf set up its own set of configure flags
		CXXFLAGS=" "
	]
)


# This should be called after the value of CXXFLAGS has settled
AC_DEFUN([SSS_FIXUP_COMPILER_FLAGS],
	[
		AC_LANG_PUSH(C++)
		
		AC_CHECK_COMPILER_FLAGS([-Wall -Wextra])
		
		autoconf_calculated_cxxflags="${CXXFLAGS}"
		
		# Fill in DEBUG_CXXFLAGS
		if test x"${DEBUG_CXXFLAGS}" != x"" ; then
			DEBUG_CXXFLAGS="${autoconf_calculated_cxxflags} ${DEBUG_CXXFLAGS}"
		else
			#AC_CHECK_FIRST_COMPILER_FLAG([-Og -O0])
			AC_CHECK_COMPILER_FLAG([-O0])
			AC_CHECK_FIRST_COMPILER_FLAG([-g3 -g2 -g])
			DEBUG_CXXFLAGS="${CXXFLAGS}"
		fi
		
		# Fill in CXXFLAGS
		CXXFLAGS="${autoconf_calculated_cxxflags}"
		if test x"${user_CXXFLAGS}" != x"" ; then
			OPT_CXXFLAGS="${user_CXXFLAGS}"
		else
			AC_CHECK_FIRST_COMPILER_FLAG([-O3 -O2 -O])
			AC_CHECK_COMPILER_FLAG([-flto])
			OPT_CXXFLAGS="${CXXFLAGS}"
		fi
		
		CXXFLAGS="${autoconf_calculated_cxxflags}"
		
		AC_SUBST(DEBUG_CXXFLAGS)
		AC_SUBST(OPT_CXXFLAGS)
		
		AC_LANG_POP(C++)
	]
)

