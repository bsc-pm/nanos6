#	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
#
#	Copyright (C) 2015-2022 Barcelona Supercomputing Center (BSC)

AC_DEFUN([CONFIGURE_NANOS6_FEATURES],
	[
		ac_nanos6_suports_cpu_management=yes
		ac_nanos6_supports_user_mutex=yes
		NANOS6_VARIANT=optimized

		_CONFIGURE_NANOS6_FEATURES
	]
)

AC_DEFUN([CONFIGURE_NANOS6_ARGOBOTS_FEATURES],
	[
		ac_nanos6_suports_cpu_management=no
		ac_nanos6_supports_user_mutex=no
		NANOS6_VARIANT=argobots

		_CONFIGURE_NANOS6_FEATURES
	]
)

AC_DEFUN([_CONFIGURE_NANOS6_FEATURES],
	[
		AM_CONDITIONAL(HAVE_CPU_MANAGEMENT, test x"${ac_nanos6_suports_cpu_management}" = x"yes")
		AM_CONDITIONAL(HAVE_WORKING_USER_MUTEX, test x"${ac_nanos6_supports_user_mutex}" = x"yes")
		AC_SUBST([NANOS6_VARIANT])
	]
)

AC_DEFUN([SELECT_NANOS6_INSTRUMENTATIONS],
	[
		AC_MSG_CHECKING([whether to build instrumentation variants])
		AC_ARG_ENABLE(
			[all-instrumentations],
			[AS_HELP_STRING([--disable-all-instrumentations], [do not build any instrumented variant. However, the individual options to enable/disable specific instrumentations always override the behavior of this option regarding the corresponding instrumentation variant])],
			[
				case "${enableval}" in
				yes)
					ac_build_all_instrumentations=yes
					;;
				no)
					ac_build_all_instrumentations=no
					;;
				*)
					AC_MSG_ERROR([bad value ${enableval} for --enable-all-instrumentations])
					;;
				esac
			],
			[ac_build_all_instrumentations=yes]
		)
		AC_MSG_RESULT([$ac_build_all_instrumentations])

		if test x"${ac_build_all_instrumentations}" = x"yes"; then
			ac_build_ctf_instrumentation=yes
			ac_build_extrae_instrumentation=yes
			ac_build_graph_instrumentation=yes
			ac_build_lint_instrumentation=yes
			ac_build_stats_instrumentation=yes
			ac_build_verbose_instrumentation=yes
		else
			ac_build_ctf_instrumentation=no
			ac_build_extrae_instrumentation=no
			ac_build_graph_instrumentation=no
			ac_build_lint_instrumentation=no
			ac_build_stats_instrumentation=no
			ac_build_verbose_instrumentation=no
		fi

		AC_MSG_CHECKING([whether to build the ctf instrumented variant])
		AC_ARG_ENABLE(
			[ctf-instrumentation],
			[AS_HELP_STRING([--disable-ctf-instrumentation], [build the ctf instrumented variant])],
			[
				case "${enableval}" in
				yes)
					ac_build_ctf_instrumentation=yes
					;;
				no)
					ac_build_ctf_instrumentation=no
					;;
				*)
					AC_MSG_ERROR([bad value ${enableval} for --enable-ctf-instrumentation])
					;;
				esac
			], []
		)
		AC_MSG_RESULT([$ac_build_ctf_instrumentation])
		AM_CONDITIONAL(BUILD_CTF_INSTRUMENTATION, test x"${ac_build_ctf_instrumentation}" = x"yes")

		AC_MSG_CHECKING([whether to build the extrae instrumented variant])
		AC_ARG_ENABLE(
			[extrae-instrumentation],
			[AS_HELP_STRING([--disable-extrae-instrumentation], [build the extrae instrumented variant])],
			[
				case "${enableval}" in
				yes)
					ac_build_extrae_instrumentation=yes
					;;
				no)
					ac_build_extrae_instrumentation=no
					;;
				*)
					AC_MSG_ERROR([bad value ${enableval} for --enable-extrae-instrumentation])
					;;
				esac
			], []
		)
		AC_MSG_RESULT([$ac_build_extrae_instrumentation])
		AM_CONDITIONAL(BUILD_EXTRAE_INSTRUMENTATION, test x"${ac_build_extrae_instrumentation}" = x"yes")

		AC_MSG_CHECKING([whether to build the graph instrumented variant])
		AC_ARG_ENABLE(
			[graph-instrumentation],
			[AS_HELP_STRING([--disable-graph-instrumentation], [build the graph instrumented variant])],
			[
				case "${enableval}" in
				yes)
					ac_build_graph_instrumentation=yes
					;;
				no)
					ac_build_graph_instrumentation=no
					;;
				*)
					AC_MSG_ERROR([bad value ${enableval} for --enable-graph-instrumentation])
					;;
				esac
			], []
		)
		AC_MSG_RESULT([$ac_build_graph_instrumentation])
		AM_CONDITIONAL(BUILD_GRAPH_INSTRUMENTATION, test x"${ac_build_graph_instrumentation}" = x"yes")

		AC_MSG_CHECKING([whether to build the lint instrumented variant])
		AC_ARG_ENABLE(
			[lint-instrumentation],
			[AS_HELP_STRING([--disable-lint-instrumentation], [build the lint instrumented variant])],
			[
				case "${enableval}" in
				yes)
					ac_build_lint_instrumentation=yes
					;;
				no)
					ac_build_lint_instrumentation=no
					;;
				*)
					AC_MSG_ERROR([bad value ${enableval} for --enable-lint-instrumentation])
					;;
				esac
			], []
		)
		AC_MSG_RESULT([$ac_build_lint_instrumentation])
		AM_CONDITIONAL(BUILD_LINT_INSTRUMENTATION, test x"${ac_build_lint_instrumentation}" = x"yes")

		AC_MSG_CHECKING([whether to build the stats instrumented variant])
		AC_ARG_ENABLE(
			[stats-instrumentation],
			[AS_HELP_STRING([--disable-stats-instrumentation], [build the stats instrumented variant])],
			[
				case "${enableval}" in
				yes)
					ac_build_stats_instrumentation=yes
					;;
				no)
					ac_build_stats_instrumentation=no
					;;
				*)
					AC_MSG_ERROR([bad value ${enableval} for --enable-stats-instrumentation])
					;;
				esac
			], []
		)
		AC_MSG_RESULT([$ac_build_stats_instrumentation])
		AM_CONDITIONAL(BUILD_STATS_INSTRUMENTATION, test x"${ac_build_stats_instrumentation}" = x"yes")

		AC_MSG_CHECKING([whether to build the verbose instrumented variant])
		AC_ARG_ENABLE(
			[verbose-instrumentation],
			[AS_HELP_STRING([--disable-verbose-instrumentation], [build the verbose instrumented variant])],
			[
				case "${enableval}" in
				yes)
					ac_build_verbose_instrumentation=yes
					;;
				no)
					ac_build_verbose_instrumentation=no
					;;
				*)
					AC_MSG_ERROR([bad value ${enableval} for --enable-verbose-instrumentation])
					;;
				esac
			], []
		)
		AC_MSG_RESULT([$ac_build_verbose_instrumentation])
		AM_CONDITIONAL(BUILD_VERBOSE_INSTRUMENTATION, test x"${ac_build_verbose_instrumentation}" = x"yes")
	]
)

