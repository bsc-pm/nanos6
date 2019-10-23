#	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
#	
#	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)


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
				
				if test x"${USING_MERCURIUM}" = x"yes" ; then
					AC_MSG_CHECKING([if $[]_AC_CC[] $[]_AC_LANG_PREFIX[]FLAGS supports the $1 flag through the preprocessor])
					_AC_LANG_PREFIX[]FLAGS="$[]_AC_LANG_PREFIX[]FLAGS --Wp,$1"
					AC_LINK_IFELSE(
						[AC_LANG_PROGRAM([[]], [[]])],
						[
							AC_MSG_RESULT([yes])
							# Save the flag
							ac_save_[]_AC_LANG_PREFIX[]FLAGS="$[]_AC_LANG_PREFIX[]FLAGS"
						], [
							AC_MSG_RESULT([no])
							_AC_LANG_PREFIX[]FLAGS="$ac_save_[]_AC_LANG_PREFIX[]FLAGS"
						]
					)
					
					AC_MSG_CHECKING([if $[]_AC_CC[] $[]_AC_LANG_PREFIX[]FLAGS supports the $1 flag through the native compiler])
					_AC_LANG_PREFIX[]FLAGS="$[]_AC_LANG_PREFIX[]FLAGS --Wn,$1"
					AC_LINK_IFELSE(
						[AC_LANG_PROGRAM([[]], [[]])],
						[
							AC_MSG_RESULT([yes])
							# Save the flag
							ac_save_[]_AC_LANG_PREFIX[]FLAGS="$[]_AC_LANG_PREFIX[]FLAGS"
						], [
							AC_MSG_RESULT([no])
							_AC_LANG_PREFIX[]FLAGS="$ac_save_[]_AC_LANG_PREFIX[]FLAGS"
						]
					)
					
					AC_MSG_CHECKING([if $[]_AC_CC[] $[]_AC_LANG_PREFIX[]FLAGS supports the $1 flag through the linker])
					_AC_LANG_PREFIX[]FLAGS="$[]_AC_LANG_PREFIX[]FLAGS --Wl,$1"
					AC_LINK_IFELSE(
						[AC_LANG_PROGRAM([[]], [[]])],
						[
							AC_MSG_RESULT([yes])
						], [
							AC_MSG_RESULT([no])
							_AC_LANG_PREFIX[]FLAGS="$ac_save_[]_AC_LANG_PREFIX[]FLAGS"
						]
					)
				fi
			]
		)
		
	]
)


AC_DEFUN([AC_CHECK_COMPILER_FLAGS],
	[
		for check_flag in $1 ; do
			AC_CHECK_COMPILER_FLAG([$check_flag])
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


dnl AC_CHECK_EXTRACT_FIRST_COMPILER_FLAG(VARIABLE-NAME, [list of flags])
AC_DEFUN([AC_CHECK_EXTRACT_FIRST_COMPILER_FLAG],
	[
		ac_save2_[]_AC_LANG_PREFIX[]FLAGS="$[]_AC_LANG_PREFIX[]FLAGS"
		for flag in $2 ; do
			AC_CHECK_COMPILER_FLAG([$flag])
			if test x"$ac_save2_[]_AC_LANG_PREFIX[]FLAGS" != x"$[]_AC_LANG_PREFIX[]FLAGS" ; then
				$1[]="$flag"
				break;
			fi
		done
		[]_AC_LANG_PREFIX[]FLAGS="$ac_save2_[]_AC_LANG_PREFIX[]FLAGS"
		AC_SUBST($1)
	]
)


# This should be called before AC_PROG_CXX
AC_DEFUN([SSS_PREPARE_COMPILER_FLAGS],
	[
		AC_ARG_VAR(DEBUG_CXXFLAGS, [C++ compiler flags for debugging versions])
		AC_ARG_VAR(PROFILE_CXXFLAGS, [C++ compiler flags for profiling versions])
		
		user_CXXFLAGS="${CXXFLAGS}"
		# Do not let autoconf set up its own set of configure flags
		CXXFLAGS=" "
	]
)


# This should be called after the value of CXXFLAGS has settled
AC_DEFUN([SSS_FIXUP_COMPILER_FLAGS],
	[
		AC_LANG_PUSH(C++)
		
		AC_CHECK_COMPILER_FLAGS([-Wall -Wextra -Wdisabled-optimization -Wshadow -fvisibility=hidden])
		
		autoconf_calculated_cxxflags="${CXXFLAGS}"
		
		# Fill in DEBUG_CXXFLAGS
		if test x"${DEBUG_CXXFLAGS}" != x"" ; then
			DEBUG_CXXFLAGS="${autoconf_calculated_cxxflags} ${DEBUG_CXXFLAGS}"
		else
			#AC_CHECK_FIRST_COMPILER_FLAG([-Og -O0])
			AC_CHECK_COMPILER_FLAG([-O0])
			AC_CHECK_FIRST_COMPILER_FLAG([-g3 -g2 -g])
			AC_CHECK_FIRST_COMPILER_FLAG([-fstack-security-check -fstack-protector-all])
			DEBUG_CXXFLAGS="${CXXFLAGS}"
		fi
		
		if test x"${PROFILE_CXXFLAGS}" != x"" ; then
			PROFILE_CXXFLAGS="${autoconf_calculated_cxxflags} ${PROFILE_CXXFLAGS}"
		fi
		
		# Fill in CXXFLAGS
		CXXFLAGS="${autoconf_calculated_cxxflags}"
		if test x"${user_CXXFLAGS}" != x"" ; then
			OPT_CXXFLAGS="${user_CXXFLAGS}"
		else
			AC_CHECK_FIRST_COMPILER_FLAG([-O3 -O2 -O])
			if test x"${PROFILE_CXXFLAGS}" != x"" ; then
				OPT_CXXFLAGS="${CXXFLAGS}"
				AC_CHECK_FIRST_COMPILER_FLAG([-g3 -g2 -g])
				PROFILE_CXXFLAGS="${CXXFLAGS}"
				CXXFLAGS=${OPT_CXXFLAGS}
			fi
			AC_CHECK_COMPILER_FLAG([-flto])
			OPT_CXXFLAGS="${CXXFLAGS}"
		fi
		
		CXXFLAGS="${autoconf_calculated_cxxflags}"
		
		AC_SUBST(DEBUG_CXXFLAGS)
		AC_SUBST(OPT_CXXFLAGS)
		AC_SUBST(PROFILE_CXXFLAGS)
		
		AC_LANG_POP(C++)
	]
)

