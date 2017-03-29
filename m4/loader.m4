AC_DEFUN([SSS_CHECK_LOADER],
	[
		AC_ARG_WITH(
			[installed-loader],
			[AS_HELP_STRING([--with-installed-loader=prefix], [specify the prefix of the Nanos6 loader installation @<:@default=same as prefix@:>@])],
			[ac_loader_dir="${withval}"],
			[ac_loader_dir="${prefix}"]
		)
		NANOS6_HEADER_DIR="${ac_loader_dir}/include"
		AC_SUBST([NANOS6_HEADER_DIR])
		
		ac_save_cppflags="${CPPFLAGS}"
		ac_save_libs="${LIBS}"
		CPPFLAGS="${CPPFLAGS} -I${NANOS6_HEADER_DIR}"
		LIBS="${LIBS} -L${ac_loader_dir}/lib -L${ac_loader_dir}/lib64 -L${ac_loader_dir}/lib32"
		
		CHECK_LOADER_INSTALLATION_MSG="Please check that the nanos6 loader is installed either at the same prefix or use the --with-installed-loader parameter to specify an alternative path."
		
		AC_LANG_PUSH(C)
		AC_CHECK_HEADER(nanos6.h, [], [AC_MSG_ERROR([Cannot find the nanos6.h header file. ${CHECK_LOADER_INSTALLATION_MSG}])])
		unset ac_cv_header_nanos6_h
		
		AC_CHECK_HEADER(nanos6/blocking.h, [], [AC_MSG_ERROR([Cannot find the nanos6/blocking.h header file. ${CHECK_LOADER_INSTALLATION_MSG}])])
		AC_CHECK_HEADER(nanos6/bootstrap.h, [], [AC_MSG_ERROR([Cannot find the nanos6/bootstrap.h header file. ${CHECK_LOADER_INSTALLATION_MSG}])])
		AC_CHECK_HEADER(nanos6/constants.h, [], [AC_MSG_ERROR([Cannot find the nanos6/constants.h header file. ${CHECK_LOADER_INSTALLATION_MSG}])])
		AC_CHECK_HEADER(nanos6/debug.h, [], [AC_MSG_ERROR([Cannot find the nanos6/debug.h header file. ${CHECK_LOADER_INSTALLATION_MSG}])])
		AC_CHECK_HEADER(nanos6/dependencies.h, [], [AC_MSG_ERROR([Cannot find the nanos6/dependencies.h header file. ${CHECK_LOADER_INSTALLATION_MSG}])])
		AC_CHECK_HEADER(nanos6/final.h, [], [AC_MSG_ERROR([Cannot find the nanos6/final.h header file. ${CHECK_LOADER_INSTALLATION_MSG}])])
		AC_CHECK_HEADER(nanos6/library-mode.h, [], [AC_MSG_ERROR([Cannot find the nanos6/library-mode.h header file. ${CHECK_LOADER_INSTALLATION_MSG}])])
		AC_CHECK_HEADER(nanos6/multidimensional-dependencies.h, [], [AC_MSG_ERROR([Cannot find the nanos6/multidimensional-dependencies.h header file. ${CHECK_LOADER_INSTALLATION_MSG}])])
		AC_CHECK_HEADER(nanos6/multidimensional-release.h, [], [AC_MSG_ERROR([Cannot find the nanos6/multidimensional-release.h header file. ${CHECK_LOADER_INSTALLATION_MSG}])])
		AC_CHECK_HEADER(nanos6/polling.h, [], [AC_MSG_ERROR([Cannot find the nanos6/polling.h header file. ${CHECK_LOADER_INSTALLATION_MSG}])])
		AC_CHECK_HEADER(nanos6/task-instantiation.h, [], [AC_MSG_ERROR([Cannot find the nanos6/task-instantiation.h header file. ${CHECK_LOADER_INSTALLATION_MSG}])])
		AC_CHECK_HEADER(nanos6/taskwait.h, [], [AC_MSG_ERROR([Cannot find the nanos6/taskwait.h header file. ${CHECK_LOADER_INSTALLATION_MSG}])])
		AC_CHECK_HEADER(nanos6/user-mutex.h, [], [AC_MSG_ERROR([Cannot find the nanos6/user-mutex.h header file. ${CHECK_LOADER_INSTALLATION_MSG}])])
		
		AC_CHECK_LIB(nanos6, nanos_submit_task, [], [AC_MSG_ERROR([Cannot find the nanos6 loader library. ${CHECK_LOADER_INSTALLATION_MSG}])])
		AC_LANG_POP(C)
		
		CPPFLAGS="${ac_save_cppflags}"
		LIBS="${ac_save_libs}"
	]
)


