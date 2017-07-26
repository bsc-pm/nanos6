#	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
#	
#	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)

AC_DEFUN([AC_CHECK_SPECIAL_HOST_ARCH],
	[
		AC_REQUIRE([AC_CANONICAL_HOST])
		
		AC_BEFORE([$0], [AC_PROG_CPP])
		AC_BEFORE([$0], [AC_PROG_CC])
		AC_BEFORE([$0], [AC_PROG_CXX])
		
		AC_ARG_WITH(
			[k1om],
			[AS_HELP_STRING([--with-k1om=prefix], [specify the installation prefix of the k1om GNU compilers @<:@default=none@:>@])],
			[ac_cv_use_k1om_prefix="${withval}"]
		)
		
		
		if test x"${host_vendor}" = x"k1om" ; then
			AC_MSG_CHECKING([the installation prefix of the k1om GNU compilers])
			if test -z "${ac_cv_use_k1om_prefix}" ; then
				AC_MSG_RESULT([none])
			else
				AC_MSG_RESULT([${ac_cv_use_k1om_prefix}])
			fi
			
			if test -z "${ac_cv_use_k1om_prefix}" ; then
				AC_MSG_CHECKING([if the installation prefix of the k1om GNU compilers can be autodetected])
				candidate_k1om_prefix=$(which x86_64-k1om-linux-gcc | sed 's/\/bin\/x86_64-k1om-linux-gcc$//')
				if test -z "${ac_cv_use_k1om_prefix}" ; then
					candidate_k1om_prefix=$(echo /usr/linux-k1om-* | awk '{ print $N; }')
				fi
				
				if test ! -z "${candidate_k1om_prefix}" ; then
					AC_MSG_RESULT([${candidate_k1om_prefix}])
					ac_cv_use_k1om_prefix="${candidate_k1om_prefix}"
				else
					AC_MSG_RESULT([no])
					AC_MSG_ERROR([To use the MIC architecture, please also pass the --with-k1om flag. For instance --with-k1om=/usr/linux-k1om-4.7])
				fi
			fi
			
			if test -z "$CC" ; then
				AC_CHECK_TOOL([TARGET_GCC], [gcc], [], [${ac_cv_use_k1om_prefix}/bin:${PATH}])
				if ! test -z "${TARGET_GCC}" ; then
					TARGET_GCC=$(env PATH="${ac_cv_use_k1om_prefix}/bin:${PATH}" which "${TARGET_GCC}")
					AC_CHECK_TOOL([CC], [icc -mmic -gcc-name=${TARGET_GCC}], [], [${ac_cv_use_k1om_prefix}/bin:${PATH}])
				fi
			fi
			
			if test -z "$CPP" ; then
				if ! test -z "${TARGET_GCC}" ; then
					AC_CHECK_TOOL([CPP], [icc -mmic -gcc-name=${TARGET_GCC} -E], [], [${ac_cv_use_k1om_prefix}/bin:${PATH}])
				fi
			fi
			
			if test -z "$CXX" ; then
				AC_CHECK_TOOL([TARGET_GXX], [g++], [], [${ac_cv_use_k1om_prefix}/bin:${PATH}])
				if ! test -z "${TARGET_GXX}" ; then
					TARGET_GXX=$(env PATH="${ac_cv_use_k1om_prefix}/bin:${PATH}" which "${TARGET_GXX}")
					AC_CHECK_TOOL([CXX], [icpc -mmic -gxx-name=${TARGET_GXX}], [], [${ac_cv_use_k1om_prefix}/bin:${PATH}])
				fi
			fi
			
			LDFLAGS="${LDFLAGS} -lsvml -lirng -limf -lintlc"
			TEST_LOG_COMPILER='env SINK_LD_LIBRARY_PATH=$(abs_builddir)/.libs:$(SINK_LD_LIBRARY_PATH) micnativeloadex'
			AC_SUBST([TEST_LOG_COMPILER])
		fi
	]
)

