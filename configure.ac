#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.69])
AC_INIT([nanos6], [1.0], [pm-tools-users@bsc.es])
AC_CONFIG_SRCDIR([src/scheduling/Scheduler.hpp])
AC_CONFIG_AUX_DIR([config-aux])
AC_CONFIG_MACRO_DIR([common/m4])

AC_CANONICAL_BUILD
AC_CANONICAL_HOST

AM_INIT_AUTOMAKE([foreign dist-bzip2 no-dist-gzip subdir-objects -Wall])
AM_SILENT_RULES([yes])

AC_CHECK_SPECIAL_HOST_ARCH

AC_CONFIG_HEADERS([config.h])

AM_PROG_AR
LT_INIT([shared disable-static pic-only])
AC_SUBST([LIBTOOL_DEPS])

SSS_CHECK_SOURCE_VERSION

# Checks for programs.
SSS_PREPARE_COMPILER_FLAGS
AC_PROG_CXX
AX_CXX_COMPILE_STDCXX_11([], [mandatory])

SSS_CHECK_CC_VERSION
SSS_CHECK_CXX_VERSION

AC_PROG_AWK
#AC_PROG_CC
#AC_PROG_CPP
AC_PROG_INSTALL
AC_PROG_LN_S
AC_PROG_MAKE_SET
SSS_CHECK_NANOS6_MERCURIUM

DX_DOT_FEATURE(ON)
DX_INIT_DOXYGEN([nanos6_rt], [docs/Doxyfile], [docs/doxygen])
full_top_srcdir=$(readlink -f ${srcdir})
AC_SUBST([full_top_srcdir])

# Checks for libraries.
AX_BOOST_BASE([1.59], [], [ AC_MSG_ERROR([Boost library not found])])

# Checks for header files.
AC_CHECK_HEADERS([unistd.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_CHECK_HEADER_STDBOOL
# AC_C_INLINE
AC_TYPE_SIZE_T
AC_LANG_PUSH(C++)
AC_C_RESTRICT
AC_LANG_POP(C++)

SSS_PUSH_NANOS6_MERCURIUM
SSS_CHECK_SANE_CHRONO
SSS_POP_NANOS6_MERCURIUM

# Checks for library functions.
AC_CHECK_BACKTRACE
AC_CHECK_LIB([rt], [timer_create], [BACKTRACE_LIBS="${BACKTRACE_LIBS} -lrt"])


SSS_FIXUP_COMPILER_FLAGS


CONFIGURE_NANOS6_FEATURES


AC_CONFIG_FILES([
	Makefile
	docs/Doxyfile
	src/version/VersionInfo.hpp
	tests/directive_based/check-shutdown/Makefile
	tests/directive_based/tests/Makefile
])
AC_OUTPUT

