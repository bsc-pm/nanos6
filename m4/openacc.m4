#	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
#
#	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)

AC_DEFUN([AC_PGI_VERSION_SELECT],
	[
		# Check compiler version, as provided libraries change among versions:
		# match "XX.YY", e.g. 19.10, 20.7 etc. using grep regex match (extra [] because m4 neutralizes it)
		openacc_pgi_VERSION=`${PGICXX} --version | grep -Eo ['[0-9]+[.][0-9]+']`
		AC_MSG_RESULT([Detected NVIDIA/PGI version ${openacc_pgi_VERSION}])
		case $openacc_pgi_VERSION in
			19.10*)
				# For 19.10 libs were different in x86_64 and powerPC...
				case $host in
					x86*)
						ac_manual_pgi_acc_libs="-laccapi -lpgc -laccn -laccns -laccg -laccg2"
						;;
					powerpc64*)
						ac_manual_pgi_acc_libs="-laccapi -lpgc -laccn -laccns -laccg -laccg2 -lcudapgi -lpgkomp -lpgatm -lomp"
						;;
					*)
						ac_manual_pgi_acc_libs="-laccapi -lpgc -laccn -laccns -laccg -laccg2"
						;;
				esac
				;;
			#20.4*)
				# 20.4 appears to break nanos6 loader even with all libraries linked
				#ac_manual_pgi_acc_libs="-lacchost -laccdevice -lpgiman -lnvc -lnvomp -lcudadevice -lpgmath"
				#;;
			20.7*|20.9*|20.11*|21.*)
				ac_manual_pgi_acc_libs="-lacchost -laccdevice -lnvhpcman -lnvc"
				;;
			*)
				AC_MSG_WARN([The detected PGI version is not supported. Nanos6 is tested with PGI 19.10 and PGI/NVIDIA HPC-SDK 20.7 - 20.9])
				ac_openacc_pgi_unsupported=yes
				;;
		esac

		# check AC_PATH_PROG
		# manipulate to find the bin/
		# swap to lib
		if test x"${ac_openacc_pgi_unsupported}" != x"yes" ; then
			if test x"${ac_cv_use_pgi_prefix}" != x"" ; then
				# hacky way to obtain the quoted string for rpath; $(ECHO) seems to not be provided
				PGI_LIB_PATH_Q=`echo \"${ac_cv_use_pgi_prefix}/lib\"`
				openacc_LIBS="-L${ac_cv_use_pgi_prefix}/lib ${ac_manual_pgi_acc_libs}"
				openacc_LIBS="${openacc_LIBS} -Wl,-rpath,${PGI_LIB_PATH_Q}"
				openacc_h_pgi=`find ${ac_cv_use_pgi_prefix}/include -name 'openacc.h'`
			else
				# hacky way to obtain the lib path from bin and produce quoted for rpath...
				PGI_LIB_PATH=`echo $PGICXX | sed s@bin/pgc++@lib@g`
				PGI_INC_PATH=`echo $PGICXX | sed s@bin/pgc++@include@g`
				AC_MSG_RESULT([found PGI installed libraries ${PGI_LIB_PATH}])
				PGI_LIB_PATH_Q=`echo \"${PGI_LIB_PATH}\"`
				openacc_LIBS="-L${PGI_LIB_PATH} ${ac_manual_pgi_acc_libs}"
				openacc_LIBS="${openacc_LIBS} -Wl,-rpath,${PGI_LIB_PATH_Q}"
				openacc_h_pgi=`find ${PGI_INC_PATH} -name 'openacc.h'`
			fi
			if test x"${openacc_h_pgi}" != x"" ; then
				openacc_h_pgi=`echo \"${openacc_h_pgi}\"`
				ac_use_openacc=yes
			else
				AC_MSG_WARN([openacc.h not found in PGI installation. Check path correctness])
			fi
		fi		
	]
)

AC_DEFUN([AC_CHECK_PGI],
	[
		AC_ARG_WITH(
			[pgi],
			[AS_HELP_STRING([--with-pgi=prefix], [specify the installation prefix of PGI or NVIDIA HPC-SDK])],
			[ ac_cv_use_pgi_prefix=$withval ],
			[ ac_cv_use_pgi_prefix="" ]
		)

		if test x"${ac_cv_use_pgi_prefix}" != x"" ; then
			AC_MSG_CHECKING([the PGI installation prefix])
			AC_MSG_RESULT([${ac_cv_use_pgi_prefix}])
			AC_PATH_PROG([PGICXX],[pgc++],[], [ ${ac_cv_use_pgi_prefix}/bin ])
			if test x"$PGICXX" != x""; then
				AC_PGI_VERSION_SELECT
			else
				AC_MSG_WARN([pgc++ compiler executable not found in specified path. Check provided installation paths])
			fi
		fi

		# If no specific path provided, check in env/default PATH
		if test x"${ac_use_openacc}" = x"" ; then
			AC_MSG_CHECKING([the PGI compiler installation])
			AC_PATH_PROG([PGICXX],[pgc++],[not available])
			if test "$PGICXX" != "not available" ; then
				AC_PGI_VERSION_SELECT
			else
				AC_MSG_WARN([Supported PGI compiler not found. Check current PATH or provide installation path using --with-pgi])
			fi
		fi

		AC_SUBST([openacc_LIBS])
	]
)
