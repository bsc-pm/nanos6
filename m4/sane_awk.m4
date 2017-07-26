#	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
#	
#	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)

AC_DEFUN([CHECK_SANE_AWK],
	[
		AC_REQUIRE([AC_PROG_AWK])
		AC_MSG_CHECKING([if awk supports more than 32767 fields])
		if echo $(seq 1 32768) | ${AWK} '{ print $NF; }' &> /dev/null ; then
			ac_sane_awk=yes
		else
			ac_sane_awk=no
		fi
		AC_MSG_RESULT([$ac_sane_awk])
		AM_CONDITIONAL(AWK_IS_SANE, test x"$ac_sane_awk" = x"yes")
	]
)

