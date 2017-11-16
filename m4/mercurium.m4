#	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
#	
#	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)

AC_DEFUN([SSS_CHECK_MERCURIUM],
	[
		AC_ARG_WITH(
			[mercurium],
			[AS_HELP_STRING([--with-mercurium=prefix], [specify the installation prefix of the mercurium compiler @<:@default=auto@:>@])],
			[ac_use_mercurium_prefix="${withval}"],
			[ac_use_mercurium_prefix="auto"]
		)
		
		AC_LANG_PUSH([C])
		AX_COMPILER_VENDOR
		AC_LANG_POP([C])
		
		AC_LANG_PUSH([C++])
		AX_COMPILER_VENDOR
		AC_LANG_POP([C++])
		
		if test "$ax_cv_c_compiler_vendor" = "intel" ; then
			mcc_vendor_prefix=i
		elif test "$ax_cv_c_compiler_vendor" = "ibm" ; then
			mcc_vendor_prefix=xl
		fi
		
		if test "$ax_cv_cxx_compiler_vendor" = "intel" ; then
			mcxx_vendor_prefix=i
		elif test "$ax_cv_cxx_compiler_vendor" = "ibm" ; then
			mcxx_vendor_prefix=xl
		fi
		
		if test x"${ac_use_mercurium_prefix}" = x"auto" || test x"${ac_use_mercurium_prefix}" = x"yes" ; then
			AC_PATH_PROG(MCC, ${mcc_vendor_prefix}mcc, [])
			AC_PATH_PROG(MCXX, ${mcxx_vendor_prefix}mcxx, [])
			if test x"${MCC}" = x"" || test x"${MCXX}" = x"" ; then
				if test x"${ac_use_mercurium_prefix}" = x"yes"; then
					AC_MSG_ERROR([could not find Mercurium])
				else
					AC_MSG_WARN([could not find Mercurium])
				fi
			else
				ac_use_mercurium_prefix=$(echo "${MCC}" | sed 's@/bin/'${mcc_vendor_prefix}'mcc'\$'@@')
			fi
		elif test x"${ac_use_mercurium_prefix}" != x"no" ; then
			AC_PATH_PROG(MCC, ${mcc_vendor_prefix}mcc, [], [${ac_use_mercurium_prefix}/bin])
			AC_PATH_PROG(MCXX, ${mcxx_vendor_prefix}mcxx, [], [${ac_use_mercurium_prefix}/bin])
			if test x"${MCC}" = x"" || test x"${MCXX}" = x"" ; then
				AC_MSG_ERROR([could not find Mercurium])
			else
				ac_use_mercurium_prefix=$(echo "${MCC}" | sed 's@/bin/'${mcc_vendor_prefix}'mcc'\$'@@')
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
		
		AC_LANG_PUSH([C])
		AX_COMPILER_VENDOR
		AC_LANG_POP([C])
		
		AC_LANG_PUSH([C++])
		AX_COMPILER_VENDOR
		AC_LANG_POP([C++])
		
		if test "$ax_cv_c_compiler_vendor" = "intel" ; then
			mcc_vendor_prefix=i
		elif test "$ax_cv_c_compiler_vendor" = "ibm" ; then
			mcc_vendor_prefix=xl
		fi
		
		if test "$ax_cv_cxx_compiler_vendor" = "intel" ; then
			mcxx_vendor_prefix=i
		elif test "$ax_cv_cxx_compiler_vendor" = "ibm" ; then
			mcxx_vendor_prefix=xl
		fi
		
		if test x"${ac_use_nanos6_mercurium_prefix}" = x"auto" || test x"${ac_use_nanos6_mercurium_prefix}" = x"yes" ; then
			AC_PATH_PROGS(NANOS6_MCC, [${mcc_vendor_prefix}mcc.nanos6 ${mcc_vendor_prefix}mcc], [])
			AC_PATH_PROGS(NANOS6_MCXX, [${mcxx_vendor_prefix}mcxx.nanos6 ${mcc_vendor_prefix}mcxx], [])
			if test x"${NANOS6_MCC}" = x"" || test x"${NANOS6_MCXX}" = x"" ; then
				if test x"${ac_use_nanos6_mercurium_prefix}" = x"yes"; then
					AC_MSG_ERROR([could not find Nanos6 Mercurium])
				else
					AC_MSG_WARN([could not find Nanos6 Mercurium])
					ac_have_nanos6_mercurium=no
				fi
			else
				ac_use_nanos6_mercurium_prefix=$(echo "${NANOS6_MCC}" | sed 's@/bin/'${mcc_vendor_prefix}'mcc.nanos6'\$'@@;s@/bin/'${mcc_vendor_prefix}'mcc'\$'@@')
				ac_have_nanos6_mercurium=yes
			fi
		elif test x"${ac_use_nanos6_mercurium_prefix}" != x"no" ; then
			AC_PATH_PROGS(NANOS6_MCC, [${mcc_vendor_prefix}mcc.nanos6 ${mcc_vendor_prefix}mcc], [], [${ac_use_nanos6_mercurium_prefix}/bin])
			AC_PATH_PROGS(NANOS6_MCXX, [${mcxx_vendor_prefix}mcxx.nanos6 ${mcc_vendor_prefix}mcxx], [], [${ac_use_nanos6_mercurium_prefix}/bin])
			if test x"${NANOS6_MCC}" = x"" || test x"${NANOS6_MCXX}" = x"" ; then
				AC_MSG_ERROR([could not find Nanos6 Mercurium])
			else
				ac_use_nanos6_mercurium_prefix=$(echo "${NANOS6_MCC}" | sed 's@/bin/'${mcc_vendor_prefix}'mcc.nanos6'\$'@@;s@/bin/'${mcc_vendor_prefix}'mcc'\$'@@')
				ac_have_nanos6_mercurium=yes
			fi
		else
			ac_use_nanos6_mercurium_prefix=""
			ac_have_nanos6_mercurium=no
		fi
		
		AC_MSG_CHECKING([the Nanos6 Mercurium installation prefix])
		if test x"${ac_have_nanos6_mercurium}" = x"yes" ; then
			AC_MSG_RESULT([${ac_use_nanos6_mercurium_prefix}])
		else
			AC_MSG_RESULT([not found])
		fi
		
		if test x"${NANOS6_MCC}" != x"" ; then
			ac_save_CC="${CC}"
			AC_LANG_PUSH(C)
			
			AC_MSG_CHECKING([which flag enables OmpSs-2 support in Mercurium])
			OMPSS2_FLAG=none
			
			mkdir -p conftest-header-dir/nanos6
			echo 'enum nanos6_multidimensional_dependencies_api_t { nanos6_multidimensional_dependencies_api = 2 };' > conftest-header-dir/nanos6/multidimensional-dependencies.h
			echo 'enum nanos6_multidimensional_release_api_t { nanos6_multidimensional_release_api = 1 };' > conftest-header-dir/nanos6/multidimensional-release.h
			
			# Try --ompss-v2
			CC="${NANOS6_MCC} --ompss-v2 -I${srcdir}/api -Iconftest-header-dir"
			AC_COMPILE_IFELSE(
				[ AC_LANG_SOURCE( [[
#ifndef __NANOS6__
#error Not Nanos6!
#endif

int main(int argc, char ** argv) {
	return 0;
}
]]
					) ],
				[ OMPSS2_FLAG=--ompss-v2 ],
				[ ]
			)
			
			# Try --ompss-2
			CC="${NANOS6_MCC} --ompss-2 -I${srcdir}/api -Iconftest-header-dir"
			AC_COMPILE_IFELSE(
				[ AC_LANG_SOURCE( [[
#ifndef __NANOS6__
#error Not Nanos6!
#endif

int main(int argc, char ** argv) {
	return 0;
}
]]
					) ],
				[ OMPSS2_FLAG=--ompss-2 ],
				[ ]
			)
			
			rm -Rf conftest-header-dir
			
			if test x"${OMPSS2_FLAG}" != x"none" ; then
				AC_MSG_RESULT([${OMPSS2_FLAG}])
				NANOS6_MCC="${NANOS6_MCC} ${OMPSS2_FLAG}"
				NANOS6_MCXX="${NANOS6_MCXX} ${OMPSS2_FLAG}"
			else
				AC_MSG_RESULT([none])
				AC_MSG_WARN([will not use ${NANOS6_MCC} since it does not support Nanos6])
				NANOS6_MCC=""
				NANOS6_MCXX=""
				OMPSS2_FLAG=""
				unset ac_use_nanos6_mercurium_prefix
				ac_have_nanos6_mercurium=no
			fi
			
			AC_LANG_POP(C)
			CC="${ac_save_CC}"
			
		fi
		
		NANOS6_MCC_PREFIX="${ac_use_nanos6_mercurium_prefix}"
		AC_SUBST([NANOS6_MCC_PREFIX])
		AC_SUBST([NANOS6_MCC])
		AC_SUBST([NANOS6_MCXX])
		
		AM_CONDITIONAL(HAVE_NANOS6_MERCURIUM, test x"${ac_have_nanos6_mercurium}" = x"yes")
	]
)

AC_DEFUN([SSS_PUSH_NANOS6_MERCURIUM],
	[
		AC_REQUIRE([SSS_CHECK_NANOS6_MERCURIUM])
		pre_nanos6_cc="${CC}"
		pre_nanos6_cxx="${CXX}"
		pre_nanos6_cpp="${CPP}"
		pre_nanos6_cxxcpp="${CXXCPP}"
		CC="${NANOS6_MCC} $1"
		CXX="${NANOS6_MCXX} $1"
		CPP="${NANOS6_MCC} -E $1"
		CXXPP="${NANOS6_MCXX} -E $1"
		AC_MSG_NOTICE([The following checks will be performed with Mercurium])
	]
)

AC_DEFUN([SSS_POP_NANOS6_MERCURIUM],
	[
		AC_MSG_NOTICE([The following checks will no longer be performed with Mercurium])
		CC="${pre_nanos6_cc}"
		CXX="${pre_nanos6_cxx}"
		CPP="${pre_nanos6_cpp}"
		CXXPP="${pre_nanos6_cxxcpp}"
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


AC_DEFUN([SSS_CHECK_MERCURIUM_ACCEPTS_EXTERNAL_INSTALLATION],
	[
		if test x"${ac_have_nanos6_mercurium}" = x"yes" ; then
			AC_MSG_CHECKING([if Mercurium allows using an external runtime])
			AC_LANG_PUSH([C])
			ac_save_[]_AC_LANG_PREFIX[]FLAGS="$[]_AC_LANG_PREFIX[]FLAGS"
			_AC_LANG_PREFIX[]FLAGS="$[]_AC_LANG_PREFIX[]FLAGS ${OMPSS2_FLAG} --no-default-nanos6-inc"
			AC_LINK_IFELSE(
				[AC_LANG_PROGRAM([[]], [[]])],
				[ac_mercurium_supports_external_installation=no],
				[ac_mercurium_supports_external_installation=yes]
			)
			AC_LANG_POP([C])
			_AC_LANG_PREFIX[]FLAGS="$ac_save_[]_AC_LANG_PREFIX[]FLAGS"
			AC_MSG_RESULT([$ac_mercurium_supports_external_installation])
		fi
		
		AM_CONDITIONAL([MCC_SUPORTS_EXTERNAL_INSTALL], [test x"${ac_mercurium_supports_external_installation}" = x"yes"])
	]
)

