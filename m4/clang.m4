#	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
#
#	Copyright (C) 2020-2023 Barcelona Supercomputing Center (BSC)

AC_DEFUN([SSS_CHECK_NANOS6_CLANG],
	[
		AC_ARG_WITH(
			[nanos6-clang],
			[AS_HELP_STRING([--with-nanos6-clang=prefix], [specify the installation prefix of the Nanos6 Clang compiler @<:@default=auto@:>@])],
			[ac_use_nanos6_clang_prefix="${withval}"],
			[ac_use_nanos6_clang_prefix="auto"]
		)

		if test x"${ac_use_nanos6_clang_prefix}" = x"auto" || test x"${ac_use_nanos6_clang_prefix}" = x"yes" ; then
			AC_PATH_PROGS(NANOS6_CLANG, [clang], [])
			AC_PATH_PROGS(NANOS6_CLANGXX, [clang++], [])
			if test x"${NANOS6_CLANG}" = x"" || test x"${NANOS6_CLANGXX}" = x"" ; then
				if test x"${ac_use_nanos6_clang_prefix}" = x"yes"; then
					AC_MSG_ERROR([could not find Nanos6 Clang])
				else
					AC_MSG_WARN([could not find Nanos6 Clang])
					ac_have_nanos6_clang=no
				fi
			else
				ac_use_nanos6_clang_prefix=$(echo "${NANOS6_CLANG}" | sed 's@/bin/clang@@')
				ac_have_nanos6_clang=yes
			fi
		elif test x"${ac_use_nanos6_clang_prefix}" != x"no" ; then
			AC_PATH_PROGS(NANOS6_CLANG, [clang], [], [${ac_use_nanos6_clang_prefix}/bin])
			AC_PATH_PROGS(NANOS6_CLANGXX, [clang++], [], [${ac_use_nanos6_clang_prefix}/bin])
			if test x"${NANOS6_CLANG}" = x"" || test x"${NANOS6_CLANGXX}" = x"" ; then
				AC_MSG_ERROR([could not find Nanos6 Clang])
			else
				ac_use_nanos6_clang_prefix=$(echo "${NANOS6_CLANG}" | sed 's@/bin/clang@@')
				ac_have_nanos6_clang=yes
			fi
		else
			ac_use_nanos6_clang_prefix=""
			ac_have_nanos6_clang=no
		fi

		AC_MSG_CHECKING([the Nanos6 Clang installation prefix])
		if test x"${ac_have_nanos6_clang}" = x"yes" ; then
			AC_MSG_RESULT([${ac_use_nanos6_clang_prefix}])
		else
			AC_MSG_RESULT([not found])
		fi

		if test x"${NANOS6_CLANG}" != x"" ; then
			ac_save_CC="${CC}"
			AC_LANG_PUSH(C)

			AC_MSG_CHECKING([which flag enables OmpSs-2 support in Clang])
			OMPSS2_FLAG=none

			CC="${NANOS6_CLANG} -fompss-2=libnanos6"
			AC_COMPILE_IFELSE(
				[ AC_LANG_SOURCE( [[
int main(int argc, char ** argv) {
	return 0;
}
]]
					) ],
				[ OMPSS2_FLAG="-fompss-2=libnanos6" ],
				[ ]
			)

			if test x"${OMPSS2_FLAG}" != x"none" ; then
				AC_MSG_RESULT([${OMPSS2_FLAG}])
				NANOS6_CLANG="${NANOS6_CLANG} ${OMPSS2_FLAG} --gcc-toolchain=\$(subst bin/gcc,,\$(shell which gcc))"
				NANOS6_CLANGXX="${NANOS6_CLANGXX} ${OMPSS2_FLAG} --gcc-toolchain=\$(subst bin/g++,,\$(shell which g++))"
			else
				AC_MSG_RESULT([none])
				AC_MSG_WARN([will not use ${NANOS6_CLANG} since it does not support Nanos6])
				NANOS6_CLANG=""
				NANOS6_CLANGXX=""
				OMPSS2_FLAG=""
				unset ac_use_nanos6_clang_prefix
				ac_have_nanos6_clang=no
			fi

			AC_LANG_POP(C)
			CC="${ac_save_CC}"

		fi

		NANOS6_CLANG_PREFIX="${ac_use_nanos6_clang_prefix}"
		AC_SUBST([NANOS6_CLANG_PREFIX])
		AC_SUBST([NANOS6_CLANG])
		AC_SUBST([NANOS6_CLANGXX])

		AM_CONDITIONAL(HAVE_NANOS6_CLANG, test x"${ac_have_nanos6_clang}" = x"yes")
	]
)
