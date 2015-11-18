#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.69])
AC_INIT([nanos6-mercurium-tests], [1.0], [pm-tools-users@bsc.es])
AC_CONFIG_SRCDIR([cpu-activation/cpu-activation.cpp])
AC_CONFIG_AUX_DIR([config-aux])
AC_CONFIG_MACRO_DIR([../../m4])

AC_CANONICAL_BUILD
AC_CANONICAL_HOST

AM_INIT_AUTOMAKE([foreign dist-bzip2 no-dist-gzip subdir-objects -Wall])

AC_CHECK_SPECIAL_HOST_ARCH

AC_CONFIG_HEADERS([config.h])

AM_PROG_AR
LT_INIT([shared disable-static pic-only])
AC_SUBST([LIBTOOL_DEPS])

# Checks for programs.
SSS_PREPARE_COMPILER_FLAGS
AC_PROG_CC
AC_PROG_CPP
AC_PROG_CXX
SSS_CHECK_MERCURIUM
if test x"${ac_use_mercurium_prefix}" != x"" ; then
	SSS_REPLACE_WITH_MERCURIUM
fi
AX_CXX_COMPILE_STDCXX_11([], [mandatory])

AC_PROG_AWK
AC_PROG_INSTALL
AC_PROG_LN_S
AC_PROG_MAKE_SET

AC_ARG_WITH(
	[nanos6-libdir],
	[AS_HELP_STRING([--with-nanos6-libdir=libdir], [the directory where the Nanos6 runtime library is installed @<:@default=none@:>@])],
	[ac_use_nanos6_libdir="${withval}"],
	[ac_use_nanos6_libdir=""]
)
AM_CONDITIONAL([HAVE_NANOS6_LIBRARY_PATH], [test x"$ac_use_nanos6_libdir" != x""])
NANOS6_LIBRARY_PATH="$ac_use_nanos6_libdir"
AC_SUBST([NANOS6_LIBRARY_PATH])

# Checks for libraries.

# Checks for header files.
AC_CHECK_HEADERS([string.h unistd.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_CHECK_HEADER_STDBOOL
AC_TYPE_SIZE_T
AC_LANG_PUSH(C++)
AC_C_RESTRICT
AC_LANG_POP(C++)

# Checks for library functions.
AC_CHECK_FUNCS([strerror])

SSS_FIXUP_COMPILER_FLAGS

AC_CONFIG_FILES([Makefile])
AC_OUTPUT
