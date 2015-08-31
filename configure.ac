#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.69])
AC_INIT([nanos6], [1.0], [pm-tools-users@bsc.es])
AC_CONFIG_SRCDIR([src/scheduling/Scheduler.hpp])
AC_CONFIG_AUX_DIR([config-aux])
AC_CONFIG_MACRO_DIR([m4])

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
AC_PROG_CXX
AX_CXX_COMPILE_STDCXX_11([], [mandatory])

AC_PROG_AWK
#AC_PROG_CC
#AC_PROG_CPP
AC_PROG_INSTALL
AC_PROG_LN_S
AC_PROG_MAKE_SET

DX_DOT_FEATURE(ON)
DX_INIT_DOXYGEN([nanos6_rt], [docs/Doxyfile], [docs/doxygen])

# Checks for libraries.

# Checks for header files.
AC_CHECK_HEADERS([unistd.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_CHECK_HEADER_STDBOOL
# AC_C_INLINE
AC_TYPE_SIZE_T
AC_LANG_PUSH(C++)
AC_C_RESTRICT
AC_LANG_POP(C++)

# Checks for library functions.



SSS_FIXUP_COMPILER_FLAGS

AC_CONFIG_FILES([Makefile docs/Doxyfile])
AC_OUTPUT

