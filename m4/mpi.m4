AC_DEFUN([AC_ARG_MPICC],
	[
		AC_ARG_VAR(MPICC, [The MPI C compiler])
	]
)

# Just check
AC_DEFUN([AC_PREPARE_MPI],
	[
		AC_REQUIRE([AC_ARG_MPICC])
		
		AC_LANG_PUSH(C++)
		LX_FIND_MPI
		AC_LANG_POP(C++)
		
		AM_CONDITIONAL([HAVE_MPI], [test x"${have_CXX_mpi}" = x"yes"])
		AC_SUBST(MPI_CXXFLAGS)
		AC_SUBST(MPI_CXXLDFLAGS)
	]
)

# Check, set compiler and linker parameters, and fail if not found
AC_DEFUN([AC_DEMAND_MPI],
	[
		AC_REQUIRE([AC_PREPARE_MPI])
		if test "${have_CXX_mpi}" != "yes" ; then
			AC_MSG_ERROR([Could not find the MPI compiler. Please try setting the MPICC environment variable.])
		fi
	]
)

