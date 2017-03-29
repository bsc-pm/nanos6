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

