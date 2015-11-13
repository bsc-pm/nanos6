AC_DEFUN([SSS_CHECK_MERCURIUM],
	[
		AC_ARG_WITH(
			[mercurium],
			[AS_HELP_STRING([--with-mercurium=prefix], [specify the installation prefix of the mercurium compiler @<:@default=auto@:>@])],
			[ac_use_mercurium_prefix="${withval}"],
			[ac_use_mercurium_prefix="auto"]
		)
		
		if test x"${ac_use_mercurium_prefix}" = x"auto" || test x"${ac_use_mercurium_prefix}" = x"yes" ; then
			AC_PATH_PROG(MCC, mcc, [])
			AC_PATH_PROG(MCXX, mcxx, [])
			if test x"${MCC}" = x"" || test x"${MCXX}" = x"" ; then
				if test x"${ac_use_mercurium_prefix}" = x"yes"; then
					AC_MSG_ERROR([could not find Mercurium])
				else
					AC_MSG_WARN([could not find Mercurium])
				fi
			else
				ac_use_mercurium_prefix=$(echo "${MCC}" | sed 's@/bin/mcc'\$'@@')
			fi
		elif test x"${ac_use_mercurium_prefix}" != x"no" ; then
			AC_PATH_PROG(MCC, mcc, [], [${ac_use_mercurium_prefix}/bin])
			AC_PATH_PROG(MCXX, mcxx, [], [${ac_use_mercurium_prefix}/bin])
			if test x"${MCC}" = x"" || test x"${MCXX}" = x"" ; then
				AC_MSG_ERROR([could not find Mercurium])
			else
				ac_use_mercurium_prefix=$(echo "${MCC}" | sed 's@/bin/mcc'\$'@@')
			fi
		else
			ac_use_mercurium_prefix=""
		fi
		
		AC_MSG_CHECKING([the mercurium installation prefix])
		if test x"${ac_use_mercurium_prefix}" != x"" ; then
			AC_MSG_RESULT([${ac_use_mercurium_prefix}])
		else
			AC_MSG_RESULT([not found])
		fi
		MCC_PREFIX="${ac_use_mercurium_prefix}"
		AC_SUBST([MCC_PREFIX])
		
		AM_CONDITIONAL(TEST_WITH_MCC, test x"${ac_use_mercurium_prefix}" != x"")
	]
)


AC_DEFUN([SSS_CHECK_NANOS6_MERCURIUM],
	[
		AC_ARG_WITH(
			[nanos6-mercurium],
			[AS_HELP_STRING([--with-nanos6-mercurium=prefix], [specify the installation prefix of the Nanos6 Mercurium compiler @<:@default=auto@:>@])],
			[ac_use_nanos6_mercurium_prefix="${withval}"],
			[ac_use_nanos6_mercurium_prefix="auto"]
		)
		
		if test x"${ac_use_nanos6_mercurium_prefix}" = x"auto" || test x"${ac_use_nanos6_mercurium_prefix}" = x"yes" ; then
			AC_PATH_PROGS(NANOS6_MCC, [mcc.nanos6 mcc], [])
			AC_PATH_PROGS(NANOS6_MCXX, [mcxx.nanos6 mcxx], [])
			if test x"${NANOS6_MCC}" = x"" || test x"${NANOS6_MCXX}" = x"" ; then
				if test x"${ac_use_nanos6_mercurium_prefix}" = x"yes"; then
					AC_MSG_ERROR([could not find Nanos6 Mercurium])
				else
					# AC_MSG_WARN([could not find Nanos6 Mercurium])
					AC_MSG_ERROR([could not find Nanos6 Mercurium])
				fi
			else
				ac_use_nanos6_mercurium_prefix=$(echo "${NANOS6_MCC}" | sed 's@/bin/mcc.nanos6'\$'@@;s@/bin/mcc'\$'@@')
			fi
		elif test x"${ac_use_nanos6_mercurium_prefix}" != x"no" ; then
			AC_PATH_PROGS(NANOS6_MCC, [mcc.nanos6 mcc], [], [${ac_use_nanos6_mercurium_prefix}/bin])
			AC_PATH_PROGS(NANOS6_MCXX, [mcxx.nanos6 mcxx], [], [${ac_use_nanos6_mercurium_prefix}/bin])
			if test x"${NANOS6_MCC}" = x"" || test x"${NANOS6_MCXX}" = x"" ; then
				AC_MSG_ERROR([could not find Nanos6 Mercurium])
			else
				ac_use_nanos6_mercurium_prefix=$(echo "${NANOS6_MCC}" | sed 's@/bin/mcc.nanos6'\$'@@;s@/bin/mcc'\$'@@')
			fi
		else
			ac_use_nanos6_mercurium_prefix=""
		fi
		
		AC_MSG_CHECKING([the Nanos6 Mercurium installation prefix])
		if test x"${ac_use_nanos6_mercurium_prefix}" != x"" ; then
			AC_MSG_RESULT([${ac_use_nanos6_mercurium_prefix}])
		else
			AC_MSG_ERROR([not found])
		fi
		
		NANOS6_MCC_PREFIX="${ac_use_nanos6_mercurium_prefix}"
		AC_SUBST([NANOS6_MCC_PREFIX])
		AC_SUBST([NANOS6_MCC])
		AC_SUBST([NANOS6_MCXX])
	]
)


AC_DEFUN([SSS_CHECK_NANOS5_MERCURIUM],
	[
		AC_ARG_WITH(
			[nanos5-mercurium],
			[AS_HELP_STRING([--with-nanos5-mercurium=prefix], [specify the installation prefix of the Nanos5 Mercurium compiler @<:@default=auto@:>@])],
			[ac_use_nanos5_mercurium_prefix="${withval}"],
			[ac_use_nanos5_mercurium_prefix="auto"]
		)
		
		if test x"${ac_use_nanos5_mercurium_prefix}" = x"auto" || test x"${ac_use_nanos5_mercurium_prefix}" = x"yes" ; then
			AC_PATH_PROGS(NANOS5_MCC, [mcc.nanos5 mcc], [])
			AC_PATH_PROGS(NANOS5_MCXX, [mcxx.nanos5 mcxx], [])
			if test x"${NANOS5_MCC}" = x"" || test x"${NANOS5_MCXX}" = x"" ; then
				if test x"${ac_use_nanos5_mercurium_prefix}" = x"yes"; then
					AC_MSG_ERROR([could not find Nanos5 Mercurium])
				else
					# AC_MSG_WARN([could not find Nanos5 Mercurium])
					AC_MSG_ERROR([could not find Nanos5 Mercurium])
				fi
			else
				ac_use_nanos5_mercurium_prefix=$(echo "${NANOS5_MCC}" | sed 's@/bin/mcc.nanos5'\$'@@;s@/bin/mcc'\$'@@')
			fi
		elif test x"${ac_use_nanos5_mercurium_prefix}" != x"no" ; then
			AC_PATH_PROGS(NANOS5_MCC, [mcc.nanos5 mcc], [], [${ac_use_nanos5_mercurium_prefix}/bin])
			AC_PATH_PROGS(NANOS5_MCXX, [mcxx.nanos5 mcxx], [], [${ac_use_nanos5_mercurium_prefix}/bin])
			if test x"${NANOS5_MCC}" = x"" || test x"${NANOS5_MCXX}" = x"" ; then
				AC_MSG_ERROR([could not find Nanos5 Mercurium])
			else
				ac_use_nanos5_mercurium_prefix=$(echo "${NANOS5_MCC}" | sed 's@/bin/mcc.nanos5'\$'@@;s@/bin/mcc'\$'@@')
			fi
		else
			ac_use_nanos5_mercurium_prefix=""
		fi
		
		AC_MSG_CHECKING([the Nanos5 Mercurium installation prefix])
		if test x"${ac_use_nanos5_mercurium_prefix}" != x"" ; then
			AC_MSG_RESULT([${ac_use_nanos5_mercurium_prefix}])
		else
			AC_MSG_ERROR([not found])
		fi
		
		NANOS5_MCC_PREFIX="${ac_use_nanos5_mercurium_prefix}"
		AC_SUBST([NANOS5_MCC_PREFIX])
		AC_SUBST([NANOS5_MCC])
		AC_SUBST([NANOS5_MCXX])
	]
)


AC_DEFUN([SSS_REPLACE_WITH_MERCURIUM],
	[
		AC_MSG_NOTICE([Replacing the native compilers with Mercurium])
		NATIVE_CC="${CC}"
		CC="${MCC} --cc=$(echo ${NATIVE_CC} | awk '{ print '\$'1; }') --ld=$(echo ${NATIVE_CC} | awk '{ print '\$'1; }')"
		if test $(echo ${NATIVE_CC} | awk '{ print NF; }') -gt 1 ; then
			for extra_CC_param in $(echo ${NATIVE_CC} | cut -d " " -f 2-) ; do
				CC="${CC} --Wn,${extra_CC_param} --Wl,${extra_CC_param}"
			done
		fi
		
		NATIVE_CXX="${CXX}"
		CXX="${MCXX} --cxx=$(echo ${NATIVE_CXX} | awk '{ print '\$'1; }') --ld=$(echo ${NATIVE_CXX} | awk '{ print '\$'1; }')"
		if test $(echo ${NATIVE_CXX} | awk '{ print NF; }') -gt 1 ; then
			for extra_CXX_param in $(echo ${NATIVE_CXX} | cut -d " " -f 2-) ; do
				CXX="${CXX} --Wn,${extra_CXX_param} --Wl,${extra_CXX_param}"
			done
		fi
		
		NATIVE_CPP="${CPP}"
		CC="${CC} --cpp=$(echo ${NATIVE_CPP} | awk '{ print '\$'1; }')"
		CXX="${CXX} --cpp=$(echo ${NATIVE_CPP} | awk '{ print '\$'1; }')"
		if test $(echo ${NATIVE_CPP} | awk '{ print NF; }') -gt 1 ; then
			for extra_CPP_param in $(echo ${NATIVE_CPP} | cut -d " " -f 2-) ; do
				CC="${CC} --Wp,${extra_CPP_param}"
				CXX="${CXX} --Wp,${extra_CPP_param}"
			done
		fi
		
		AC_MSG_CHECKING([the Mercurium C compiler])
		AC_MSG_RESULT([${CC}])
		AC_MSG_CHECKING([the Mercurium C++ compiler])
		AC_MSG_RESULT([${CXX}])
		
		AC_SUBST([NATIVE_CC])
		AC_SUBST([NATIVE_CXX])
		AC_SUBST([NATIVE_CPP])
		
		if test x"${CC_VERSION}" != x"" ; then
			NATIVE_CC_VERSION="${CC_VERSION}"
			AC_SUBST([NATIVE_CC_VERSION])
			SSS_CHECK_CC_VERSION
		fi
		if test x"${CXX_VERSION}" != x"" ; then
			NATIVE_CXX_VERSION="${CXX_VERSION}"
			AC_SUBST([NATIVE_CXX_VERSION])
			SSS_CHECK_CXX_VERSION
		fi
		
		USING_MERCURIUM=yes
	]
)


AC_DEFUN([SSS_REPLACE_WITH_MERCURIUM_WRAPPER],
	[
		AC_MSG_NOTICE([Replacing the native compilers with a Mercurium wrapper])
		
		NATIVE_CPP="${CPP}"
		NATIVE_CC="${CC}"
		NATIVE_CXX="${CXX}"
		
		CC="${MCC}"
		CXX="${MCXX}"
		
		AC_MSG_CHECKING([the Mercurium C compiler])
		AC_MSG_RESULT([${CC}])
		AC_MSG_CHECKING([the Mercurium C++ compiler])
		AC_MSG_RESULT([${CXX}])
		
		AC_SUBST([NATIVE_CC])
		AC_SUBST([NATIVE_CXX])
		AC_SUBST([NATIVE_CPP])
		AC_SUBST([CC])
		AC_SUBST([CXX])
		
		if test x"${CC_VERSION}" != x"" ; then
			NATIVE_CC_VERSION="${CC_VERSION}"
			AC_SUBST([NATIVE_CC_VERSION])
			SSS_CHECK_CC_VERSION
		fi
		if test x"${CXX_VERSION}" != x"" ; then
			NATIVE_CXX_VERSION="${CXX_VERSION}"
			AC_SUBST([NATIVE_CXX_VERSION])
			SSS_CHECK_CXX_VERSION
		fi
		
		USING_MERCURIUM=yes
		USING_MERCURIUM_WRAPPER=yes
		
		AC_SUBST([NANOS6_MCC_CONFIG_DIR])
		AC_SUBST([NANOS5_MCC_CONFIG_DIR])
	]
)


AC_DEFUN([SSS_ALTERNATIVE_MERCURIUM_CONFIGURATION],
	[
		AC_MSG_CHECKING([the Mercurium configuration directory])
		MCC_CONFIG_DIR=$(${MCC} --print-config-dir | sed 's/.*: //')
		AC_MSG_RESULT([$MCC_CONFIG_DIR])
		AC_SUBST([MCC_CONFIG_DIR])
		
		AC_MSG_NOTICE([Creating local Mercurium configuration])
		mkdir -p mcc-config.d
		for config in $(cd "${MCC_CONFIG_DIR}"; eval 'echo *.config.*') ; do
			AC_MSG_NOTICE([Creating local Mercurium configuration file ${config}])
			# Replace the include directory and do not link automatically, since the runtime is compiled with libtool and has yet to be installed
			cat "${MCC_CONFIG_DIR}"/${config} | sed \
				's@{!nanox} linker_options = -L.*@{!nanox} linker_options = @;
				s@{!nanox,openmp}preprocessor_options = -I.*@{!nanox,openmp}preprocessor_options = -I'$(readlink -f "${srcdir}/../../src/api")' -include nanos6_rt_interface.h@;
				s@-lnanos6[[^ ]]*@@g;
				s@-Xlinker -rpath -Xlinker '"${MCC_PREFIX}/lib"'@@;
				s@-Xlinker -rpath -Xlinker '"${prefix}/lib"'@@' \
			> mcc-config.d/${config}
			LOCAL_MCC_CONFIG="${LOCAL_MCC_CONFIG} mcc-config.d/${config}"
		done
		AC_SUBST([LOCAL_MCC_CONFIG])
		
		if test x"${USING_MERCURIUM_WRAPPER}" != x"yes" ; then
			AC_MSG_CHECKING([how to select the local Mercurium configuration])
			ac_local_mercurium_profile_flags="--config-dir=${PWD}/mcc-config.d"
			AC_MSG_RESULT([${ac_local_mercurium_profile_flags}])
			
			CC="${CC} ${ac_local_mercurium_profile_flags} --profile=mcc"
			CXX="${CXX} ${ac_local_mercurium_profile_flags} --profile=mcxx"
		fi
	]
)

