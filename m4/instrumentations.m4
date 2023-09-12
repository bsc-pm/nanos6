#	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
#
#	Copyright (C) 2015-2023 Barcelona Supercomputing Center (BSC)

AC_DEFUN([SELECT_INSTRUMENTATIONS],
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
			ac_build_lint_instrumentation=yes
			ac_build_ovni_instrumentation=yes
			ac_build_verbose_instrumentation=yes
		else
			ac_build_ctf_instrumentation=no
			ac_build_extrae_instrumentation=no
			ac_build_lint_instrumentation=no
			ac_build_ovni_instrumentation=no
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

		AC_MSG_CHECKING([whether to build the ovni instrumented variant])
		AC_ARG_ENABLE(
			[ovni-instrumentation],
			[AS_HELP_STRING([--disable-ovni-instrumentation], [build the ovni instrumented variant])],
			[
				case "${enableval}" in
				yes)
					if test x"${ac_use_ovni}" != x"yes"; then
						AC_MSG_WARN([ovni instrumentation selected for build, but ovni library was not configured])
						AC_MSG_WARN([Disabling ovni instrumentation])
						ac_build_ovni_instrumentation=no
					else
						ac_build_ovni_instrumentation=yes
					fi
					;;
				no)
					ac_build_ovni_instrumentation=no
					;;
				*)
					AC_MSG_ERROR([bad value ${enableval} for --enable-ovni-instrumentation])
					;;
				esac
			], []
		)
		AC_MSG_RESULT([$ac_build_ovni_instrumentation])
		AM_CONDITIONAL(BUILD_OVNI_INSTRUMENTATION, test x"${ac_build_ovni_instrumentation}" = x"yes")

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

