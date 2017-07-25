#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.69])
AC_INIT([nanos6], [1.0], [pm-tools-users@bsc.es])
AC_CONFIG_SRCDIR([src/scheduling/Scheduler.hpp])
AC_CONFIG_AUX_DIR([config-aux])
AC_CONFIG_MACRO_DIR([m4])

AC_CANONICAL_BUILD
AC_CANONICAL_HOST

AM_INIT_AUTOMAKE([foreign dist-bzip2 no-dist-gzip subdir-objects -Wall tar-pax])
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
AC_PROG_CC
AC_PROG_CC_C99
AM_PROG_CC_C_O
AC_PROG_INSTALL
AC_PROG_LN_S
AC_PROG_MAKE_SET
CHECK_SANE_AWK
AC_PATH_PROG([PERL],[perl])
SSS_CHECK_NANOS6_MERCURIUM

SSS_PUSH_NANOS6_MERCURIUM
SSS_CHECK_MERCURIUM_ACCEPTS_EXTERNAL_INSTALLATION
SSS_POP_NANOS6_MERCURIUM

DX_DOT_FEATURE(ON)
DX_INIT_DOXYGEN([nanos6_rt], [docs/Doxyfile], [docs/doxygen])
full_top_srcdir=$(readlink -f ${srcdir})
AC_SUBST([full_top_srcdir])

# Checks for libraries.
AX_BOOST_BASE([1.59], [], [ AC_MSG_ERROR([Boost library not found])])
CHECK_PTHREAD

# Checks for header files.
AC_CHECK_HEADERS([stdlib.h unistd.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_CHECK_HEADER_STDBOOL
# AC_C_INLINE
AC_TYPE_SIZE_T
AC_LANG_PUSH(C++)
AC_C_RESTRICT
AC_LANG_POP(C++)


if test x"${ac_have_nanos6_mercurium}" = x"yes" ; then
	NANOS6_MCXX="${NANOS6_MCXX} -std=c++11"
fi

# Checks for library functions.
AC_CHECK_DLOPEN
AC_CHECK_BACKTRACE
AC_CHECK_PAPI
AC_CHECK_LIBNUMA

AC_CHECK_FUNCS([reallocarray aligned_alloc])

AC_CHECK_LIB([rt], [clock_gettime], [CLOCK_LIBS="${CLOCK_LIBS} -lrt"])
AC_SUBST(CLOCK_LIBS)

AC_CHECK_LIB([log], [__android_log_print], [ANDROID_LOG_LIBS="${ANDROID_LOG_LIBS} -llog"])
AC_SUBST(ANDROID_LOG_LIBS)

PKG_CHECK_MODULES([elfutils], [libdw >= 0.164],
   [AC_DEFINE([HAVE_LIBDW], [1], [Use elfutils libdw])], [AC_MSG_WARN([The profiling instrumenter is more accurate when the headers and libraries of elfutils are available.])]
)

CHECK_UNDEFINED_SYMBOL_VERIFICATION_FLAGS


SSS_FIXUP_COMPILER_FLAGS


CONFIGURE_NANOS6_FEATURES

# Check for gethostid
AC_CHECK_FUNCS([gethostid])

# Dependency engine to use
AC_ARG_WITH(
	[dependencies],
	[AS_HELP_STRING([--with-dependencies=type], [specify the type of dependencies to handle @<:@default=linear-regions-fragmented@:>@])],
	[ac_with_dependencies="${withval}"],
	[ac_with_dependencies="linear-regions-fragmented"]
)
AC_MSG_CHECKING([the type of dependencies to handle])
case x"${ac_with_dependencies}" in
	x"discrete")
		AC_MSG_RESULT([${ac_with_dependencies}])
		ac_supports_concurrent=yes
		ac_supports_reductions=yes
		;;
	x"linear-regions-unfragmented")
		AC_MSG_RESULT([${ac_with_dependencies}])
		;;
	x"linear-regions-fragmented")
		AC_MSG_RESULT([${ac_with_dependencies}])
		ac_supports_concurrent=yes
		ac_supports_reductions=yes
		;;
	x"linear-regions-alternative")
		AC_MSG_RESULT([${ac_with_dependencies}])
		ac_supports_concurrent=yes
		ac_supports_reductions=yes
		;;
	*)
		AC_MSG_ERROR([unknown dependency type ${ac_with_dependencies}])
		;;
esac
AM_CONDITIONAL([DISCRETE_DEPENDENCIES], [test "x${ac_with_dependencies}" = "xdiscrete"])
AM_CONDITIONAL([LINEAR_REGION_UNFRAGMENTED_DEPENDENCIES], [test "x${ac_with_dependencies}" = "xlinear-regions-unfragmented"])
AM_CONDITIONAL([LINEAR_REGION_FRAGMENTED_DEPENDENCIES], [test "x${ac_with_dependencies}" = "xlinear-regions-fragmented"])
AM_CONDITIONAL([LINEAR_REGION_ALTERNATIVE_DEPENDENCIES], [test "x${ac_with_dependencies}" = "xlinear-regions-alternative"])
AM_CONDITIONAL([HAVE_LINEAR_REGION_SUPPORT], [test "x${ac_with_dependencies}" != "xdiscrete"])
AM_CONDITIONAL([HAVE_CONCURRENT_SUPPORT], [test x"${ac_supports_concurrent}" = x"yes"])
AM_CONDITIONAL([HAVE_REDUCTIONS_SUPPORT], [test x"${ac_supports_reductions}" = x"yes"])


AC_ARG_ENABLE(
	[dependency-delayed-operations],
	[AS_HELP_STRING([--disable-dependency-delayed-operations], [do not delay dependency update operations])],
	[
		case "${enableval}" in
		yes)
			ac_dep_delayed_ops=yes
			;;
		no)
			ac_dep_delayed_ops=no
			;;
		*)
			AC_MSG_ERROR([bad value ${enableval} for --disable-dependency-delayed-operations])
			;;
		esac
	],
	[ac_dep_delayed_ops=yes]
)
if test x"${ac_dep_delayed_ops}" = x"no" ; then
	AC_DEFINE([NO_DEPENDENCY_DELAYED_OPERATIONS], 1, [do not delay dependency updates])
else
	AC_DEFINE([NO_DEPENDENCY_DELAYED_OPERATIONS], 0, [do not delay dependency updates])
fi


AC_MSG_CHECKING([if the runtime must embed any code changes])
AC_ARG_ENABLE(
	[embed-code-changes],
	[AS_HELP_STRING([--enable-embed-code-changes], [embed into the runtime any code changes present in the source])],
	[
		case "${enableval}" in
		yes)
			ac_embed_code_changes=yes
			;;
		no)
			ac_embed_code_changes=no
			;;
		*)
			AC_MSG_ERROR([bad value ${enableval} for --enable-embed-code-changes])
			;;
		esac
	],
	[ac_embed_code_changes=no]
)
AC_MSG_RESULT([$ac_embed_code_changes])
if test x"${ac_embed_code_changes}" = x"yes"; then
	if test x"${ac_source_in_git}" != x"yes" ; then
		AC_MSG_WARN([cannot embed code changes if source is not in a git repository])
		ac_embed_code_changes=no
	fi
fi
AM_CONDITIONAL([EMBED_CODE_CHANGES], [test x"${ac_embed_code_changes}" = x"yes"])


AC_CHECK_MAIN_WRAPPER_TYPE
AC_CHECK_SYMBOL_RESOLUTION_STRATEGY

AX_CHECK_EXTRAE
PKG_CHECK_MODULES([hwloc], [hwloc])

AM_EXTRA_RECURSIVE_TARGETS([build-tests])

AC_CONFIG_FILES([
	Makefile
	docs/Doxyfile
	tests/directive_based/Makefile
])
AC_OUTPUT


if test x"${ac_have_nanos6_mercurium}" = x"no" ; then
	AC_MSG_WARN([the directive based tests will not be checked since there is no Mercurium compiler available])
fi

