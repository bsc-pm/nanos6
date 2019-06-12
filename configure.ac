#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

#	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
#	
#	Copyright (C) 2015-2019 Barcelona Supercomputing Center (BSC)

AC_PREREQ([2.69])
AC_INIT([nanos6], [2.2], [pm-tools@bsc.es])
AC_CONFIG_SRCDIR([src/scheduling/Scheduler.hpp])
AC_CONFIG_AUX_DIR([config-aux])
AC_CONFIG_MACRO_DIR([m4])

AC_CANONICAL_BUILD
AC_CANONICAL_HOST

AM_INIT_AUTOMAKE([foreign dist-bzip2 no-dist-gzip subdir-objects -Wall tar-pax -Wno-portability])
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
	NANOS6_MCXX="${NANOS6_MCXX} -std=c++03"
fi

# Checks for library functions.
AC_CHECK_DLOPEN
AC_CHECK_BACKTRACE
AC_CHECK_PAPI
AC_CHECK_LIBNUMA
AC_CHECK_MEMKIND
AC_CHECK_PQOS

AC_CHECK_FUNCS([reallocarray aligned_alloc])
AC_CHECK_MADV_FREE

AC_CHECK_LIB([rt], [clock_gettime], [CLOCK_LIBS="${CLOCK_LIBS} -lrt"])
AC_SUBST(CLOCK_LIBS)

AC_CHECK_LIB([log], [__android_log_print], [ANDROID_LOG_LIBS="${ANDROID_LOG_LIBS} -llog"])
AC_SUBST(ANDROID_LOG_LIBS)

PKG_CHECK_MODULES([elfutils], [libdw >= 0.164],
   [
		AC_DEFINE([HAVE_LIBDW], [1], [Use elfutils libdw])
		ac_have_elfutils=yes
	], [
		AC_MSG_WARN([The profiling instrumenter is more accurate and efficient when the headers and libraries of elfutils are available.])
		ac_have_elfutils=no
	]
)
AM_CONDITIONAL([HAVE_ELFUTILS], [test "x${ac_have_elfutils}" = "xyes"])

CHECK_UNDEFINED_SYMBOL_VERIFICATION_FLAGS
CHECK_AS_NEEDED_FLAGS
ADD_AS_NEEDED_SUPPORT_TO_LIBTOOL

SSS_FIXUP_COMPILER_FLAGS

AC_CHECK_EXTRACT_FIRST_COMPILER_FLAG([MCMODEL_FLAGS], [-mcmodel=large -mcmodel=medium])

CONFIGURE_NANOS6_FEATURES
SELECT_NANOS6_INSTRUMENTATION_VARIANTS

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
	x"linear-regions-fragmented")
		AC_MSG_RESULT([${ac_with_dependencies}])
		ac_supports_concurrent=yes
		ac_supports_reductions=yes
		;;
	*)
		AC_MSG_ERROR([unknown dependency type ${ac_with_dependencies}])
		;;
esac
AM_CONDITIONAL([LINEAR_REGION_FRAGMENTED_DEPENDENCIES], [test "x${ac_with_dependencies}" = "xlinear-regions-fragmented"])
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
	if test x"${ac_source_in_git}" != x"true" ; then
		AC_MSG_WARN([cannot embed code changes if source is not in a git repository])
		ac_embed_code_changes=no
	fi
fi
AM_CONDITIONAL([EMBED_CODE_CHANGES], [test x"${ac_embed_code_changes}" = x"yes"])


AC_ARG_ENABLE(
	[less-test-threads],
	[AS_HELP_STRING([--enable-less-test-threads], [reduce the size of tests that create additional threads])],
	[
		case "${enableval}" in
		yes)
			ac_less_test_threads=yes
			;;
		no)
			ac_less_test_threads=no
			;;
		*)
			AC_MSG_ERROR([bad value ${enableval} for --enable-less-test-threads])
			;;
		esac
	],
	[ac_less_test_threads=no]
)
AM_CONDITIONAL([LESS_TEST_THREADS], [test x"${ac_less_test_threads}" = x"yes"])

AC_ARG_ENABLE([cuda], [AS_HELP_STRING([--enable-cuda], [Enable CUDA task support])])


if test x"${enable_cuda}" = x"yes" ; then
	PKG_CHECK_MODULES(
		[CUDA],
		[cudart-9.2],
		[
			ac_use_cuda=yes
		],
		[
			PKG_CHECK_MODULES(
				[CUDA],
				[cudart-9.1],
				[
					ac_use_cuda=yes
				],
				[
					PKG_CHECK_MODULES(
						[CUDA],
						[cudart-9.0],
						[
							ac_use_cuda=yes
						],
						[
							PKG_CHECK_MODULES(
								[CUDA],
								[cudart-8.0],
								[
									ac_use_cuda=yes
								],
								[
									ac_use_cuda=no
								]
							)
						]
					)
				]
			)
		]
	)
else
	ac_use_cuda=no
fi
AM_CONDITIONAL([USE_CUDA], [test x${ac_use_cuda} = x"yes"])

if test x"${ac_use_cuda}" = x"yes" ; then
	AC_DEFINE([USE_CUDA], [1], [Define if CUDA is enabled.])
fi

# Check support for Cluster
AC_ARG_ENABLE([cluster],
	[AS_HELP_STRING([--enable-cluster],
		[Enable OmpSs@Cluster support. This requires an MPI version that
		 supports MPI_THREAD_MULTIPLE to be present in your environment.
		 It implies --enable-execution-workflow])
	])

if test x"${enable_cluster}" = x"yes" ; then
	ac_use_cluster=yes
	AC_DEMAND_MPI
else
	ac_use_cluster=no
fi
AM_CONDITIONAL([USE_CLUSTER], [test x${ac_use_cluster} = x"yes"])
AM_CONDITIONAL([HAVE_MPI], [test x${ac_use_cluster} = x"yes"])

if test x"${ac_use_cluster}" = x"yes" ; then
	AC_DEFINE([USE_CLUSTER], [1], [Define if Cluster is enabled.])
fi

# Check if we should enable Execution Workflow support
AC_ARG_ENABLE(
	[execution-workflow],
	[AS_HELP_STRING([--enable-execution-workflow], [Enable execution using the Execution Workflow])]
)
if test x"${enable_execution_workflow}" = x"yes" ; then
	ac_use_workflow=yes
else
	# We need to check whether any of the devices that use
	# the workflow are enabled. The check for enabling these devices
	# or not *NEEDS* to appear before these checks
	if test x"${ac_use_cluster}" = x"yes" ; then
		ac_use_workflow=yes
	else
		ac_use_workflow=no
	fi
fi

AM_CONDITIONAL([USE_EXEC_WORKFLOW], [test x"${ac_use_workflow}" = x"yes"])
if test x"${ac_use_workflow}" = x"yes" ; then
	AC_DEFINE([USE_EXEC_WORKFLOW], [1], [Define if Execution Workflow feature is enabled.])
fi

AC_ARG_ENABLE(
	[monitoring],
	[AS_HELP_STRING([--enable-monitoring], [enable monitoring and prediction techniques of statistics])],
	[
		case "${enableval}" in
		yes)
			ac_monitoring=yes
			;;
		no)
			ac_monitoring=no
			;;
		*)
			AC_MSG_ERROR([bad value ${enableval} for --enable-monitoring])
			;;
		esac
	],
	[ac_monitoring=no]
)
AM_CONDITIONAL([MONITORING_ENABLED], [test x"${ac_monitoring}" = x"yes"])
if test x"${ac_monitoring}" = x"yes" ; then
	AC_DEFINE([MONITORING_ENABLED], [1], [Define if Monitoring is enabled.])
fi

AC_ARG_ENABLE(
	[chrono-arch],
	[AS_HELP_STRING([--enable-chrono-arch], [use an architecture-dependent chrono for timing instead of the default one (std)])],
	[
		case "${enableval}" in
		yes)
			ac_chrono_arch=yes
			;;
		no)
			ac_chrono_arch=no
			;;
		*)
			AC_MSG_ERROR([bad value ${enableval} for --enable-chrono-arch])
			;;
		esac
	],
	[ac_chrono_arch=no]
)
AM_CONDITIONAL([CHRONO_ARCH], [test x"${ac_chrono_arch}" = x"yes"])
if test x"${ac_chrono_arch}" = x"yes" ; then
	AC_DEFINE([CHRONO_ARCH], [1], [Define if the architecture-dependent chrono is used.])
fi

AC_ARG_ENABLE(
	[monitoring-hwevents],
	[AS_HELP_STRING([--enable-monitoring-hwevents], [enable monitoring and prediction techniques for hardware events])],
	[
		case "${enableval}" in
		yes)
			ac_monitoring_hwevents=yes
			;;
		no)
			ac_monitoring_hwevents=no
			;;
		*)
			AC_MSG_ERROR([bad value ${enableval} for --enable-monitoring-hwevents])
			;;
		esac
	],
	[ac_monitoring_hwevents=no]
)
AM_CONDITIONAL([MONITORING_HWEVENTS_ENABLED], [test x"${ac_monitoring_hwevents}" = x"yes"])
if test x"${ac_monitoring_hwevents}" = x"yes" ; then
	AC_DEFINE([MONITORING_HWEVENTS_ENABLED], [1], [Define if Monitoring for Hardware Events is enabled.])
fi

AC_CHECK_MAIN_WRAPPER_TYPE
AC_CHECK_SYMBOL_RESOLUTION_STRATEGY

AX_CHECK_EXTRAE
PKG_CHECK_MODULES([hwloc], [hwloc])

AM_EXTRA_RECURSIVE_TARGETS([build-tests])

AC_CONFIG_FILES([
	Makefile
	docs/Doxyfile
	commands/Makefile
	tests/directive_based/Makefile
	scripts/Makefile
])
AC_OUTPUT


if test x"${ac_have_nanos6_mercurium}" = x"no" ; then
	AC_MSG_WARN([the directive based tests will not be checked since there is no Mercurium compiler available])
fi


_AS_ECHO([])
_AS_ECHO([])
_AS_ECHO([Configuration summary:])
_AS_ECHO([   Compiler version... ${CXX_VERSION}])
_AS_ECHO([   Compiler optimization flags... ${OPT_CXXFLAGS}])
_AS_ECHO([   Compiler debug flags... ${DEBUG_CXXFLAGS}])
_AS_ECHO([])
_AS_ECHO([   Dependency implementation... ${ac_with_dependencies}])
_AS_ECHO([   Symbol resolution method... ${ac_cv_use_symbol_resolution}])
_AS_ECHO([])
_AS_ECHO([   Mercurium prefix... ${NANOS6_MCC_PREFIX}])
_AS_ECHO([])
_AS_ECHO([   Code model flags... ${MCMODEL_FLAGS}])
_AS_ECHO([])
_AS_ECHO([   Boost CPPFLAGS... ${BOOST_CPPFLAGS}])
_AS_ECHO([   Boost LDFLAGS... ${BOOST_LDFLAGS}])
_AS_ECHO([])
_AS_ECHO([   Backtrace LIBS... ${BACKTRACE_LIBS}])
_AS_ECHO([])
_AS_ECHO([   libnuma CPPFLAGS... ${libnuma_CPPFLAGS}])
_AS_ECHO([   libnuma LIBS... ${libnuma_LIBS}])
_AS_ECHO([])
_AS_ECHO([   hwloc CPPFLAGS... ${hwloc_CFLAGS}])
_AS_ECHO([   hwloc LIBS... ${hwloc_LIBS}])
_AS_ECHO([])
_AS_ECHO([   clock LIBS... ${CLOCK_LIBS}])

_AS_ECHO([])
_AS_ECHO_N([   Extrae is enabled... ])
if test x"${have_extrae}" = x"yes" ; then
	AC_MSG_RESULT([yes])
else
	AC_MSG_RESULT([no])
fi

_AS_ECHO([])
_AS_ECHO_N([   PAPI is enabled... ])
if test x"${ac_use_papi}" = x"yes" ; then
	AC_MSG_RESULT([yes])
	_AS_ECHO([   PAPI CPPFLAGS... ${papi_CPPFLAGS}])
	_AS_ECHO([   PAPI LIBS... ${papi_LIBS}])
else
	AC_MSG_RESULT([no])
fi

_AS_ECHO([])
_AS_ECHO([   elfutils CPPFLAGS... ${elfutils_CFLAGS}])
_AS_ECHO([   elfutils LIBS... ${elfutils_LIBS}])

_AS_ECHO([])
_AS_ECHO_N([   CUDA is enabled... ])
if test x"${ac_use_cuda}" = x"yes" ; then
	AC_MSG_RESULT([yes])
	_AS_ECHO([	CUDA CFLAGS... ${CUDA_CFLAGS}])
	_AS_ECHO([	CUDA LIBS... ${CUDA_LIBS}])
else
	AC_MSG_RESULT([no])
fi

_AS_ECHO([])
_AS_ECHO_N([   Cluster support is enabled... ])
if test x"${ac_use_cluster}" = x"yes" ; then
	AC_MSG_RESULT([yes])
	_AS_ECHO([   MPI CXXFLAGS... ${MPI_CXXFLAGS}])
	_AS_ECHO([   MPI CXXLDFLAGS... ${MPI_CXXLDFLAGS}])
else
	AC_MSG_RESULT([no])
fi

_AS_ECHO([])
_AS_ECHO_N([   Execution Workflow feature enabled... ])
if test x"${ac_use_workflow}" = x"yes" ; then
	AC_MSG_RESULT([yes])
else
	AC_MSG_RESULT([no])
fi

_AS_ECHO([])
_AS_ECHO_N([   Memkind is enabled... ])
if test x"${ac_use_memkind}" = x"yes" ; then
	AC_MSG_RESULT([yes])
	_AS_ECHO([	Memkind CPPFLAGS... ${memkind_CPPFLAGS}])
	_AS_ECHO([	Memkind LIBS... ${memkind_LIBS}])
else
	AC_MSG_RESULT([no])
fi

_AS_ECHO([])
_AS_ECHO_N([   Monitoring is enabled... ])
if test x"${ac_monitoring}" = x"yes" ; then
	AC_MSG_RESULT([yes])
else
	AC_MSG_RESULT([no])
fi

_AS_ECHO([])
_AS_ECHO_N([   Chrono Arch is enabled... ])
if test x"${ac_chrono_arch}" = x"yes" ; then
	AC_MSG_RESULT([yes])
else
	AC_MSG_RESULT([no])
fi

_AS_ECHO([])
_AS_ECHO_N([   PQOS is enabled... ])
if test x"${ac_use_pqos}" = x"yes" ; then
	AC_MSG_RESULT([yes])
	_AS_ECHO([   PQOS CPPFLAGS... ${pqos_CPPFLAGS}])
	_AS_ECHO([   PQOS LIBS... ${pqos_LIBS}])
	if test x"${ac_use_papi}" = x"yes" ; then
		AC_MSG_WARN([Both PQoS hardware event monitoring and PAPI Instrumentation are enabled.])
		AC_MSG_WARN([Using both at the same time may cause conflicts when polling events.])
	fi
else
	AC_MSG_RESULT([no])
fi

