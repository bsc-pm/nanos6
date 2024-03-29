#	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
#
#	Copyright (C) 2015-2022 Barcelona Supercomputing Center (BSC)

AC_DEFUN([AC_CHECK_CUDA],
	[
		AC_ARG_WITH(
			[cuda],
			[AS_HELP_STRING([--with-cuda, --with-cuda=prefix], [enable CUDA; optionally specify the installation prefix of CUDA])],
			[ ac_cv_use_cuda_prefix=$withval;
			  ac_use_cuda="yes" ],
			[ ac_use_cuda="no" ]
		)

		if test x"${ac_use_cuda}" != x"no" ;then
			if test x"${ac_cv_use_cuda_prefix}" != x"yes" ; then
				AC_MSG_CHECKING([the CUDA installation prefix])
				AC_MSG_RESULT([${ac_cv_use_cuda_prefix}])
				# hacky way to obtain the quoted string for rpath
				cuda_lib_path_q=`echo \"${ac_cv_use_cuda_prefix}/lib64\"`

				AC_CHECK_FILE([${cuda_lib_path_q}/libOpenCL.so], [
					opencl_libs=-lOpenCL
					ac_use_cuda_cl="yes"
				], [
					ac_use_cuda_cl="no"
				])

				CUDA_LIBS="-L${ac_cv_use_cuda_prefix}/lib64 -lcudart -lcuda -lnvrtc ${opencl_libs}"
				CUDA_LIBS="${CUDA_LIBS} -Wl,-rpath,${cuda_lib_path_q}"
				CUDA_CFLAGS="-I${ac_cv_use_cuda_prefix}/include"
				ac_use_cuda=yes
			else
				PKG_CHECK_MODULES([CUDA], [cuda-10.2 cudart-10.2 nvrtc-10.2], [ac_use_cuda=yes], [
				PKG_CHECK_MODULES([CUDA], [cuda-10.1 cudart-10.1 nvrtc-10.1], [ac_use_cuda=yes], [
				PKG_CHECK_MODULES([CUDA], [cuda-10.0 cudart-10.0 nvrtc-10.0], [ac_use_cuda=yes], [
				PKG_CHECK_MODULES([CUDA],  [cuda-9.2 cudart-9.2 nvrtc-9.2], [ac_use_cuda=yes], [
				PKG_CHECK_MODULES([CUDA],  [cuda-9.1 cudart-9.1 nvrtc-9.1], [ac_use_cuda=yes], [
				PKG_CHECK_MODULES([CUDA],  [cuda-9.0 cudart-9.0 nvrtc-9.0], [ac_use_cuda=yes], [
				PKG_CHECK_MODULES([CUDA],  [cuda-8.0 cudart-8.0 nvrtc-8.0], [ac_use_cuda=yes], [ac_use_cuda=no]
					)])])])])])])
			fi

			if test x"${ac_use_cuda}" != x"yes" ; then
				AC_CHECK_HEADERS([cuda.h, cuda_runtime_api.h])
				# check support of all used runtime calls in CUDA library;
				# spaces mangled for output consistency.
				AC_CHECK_LIB([cudart],
	    [cudaSetDevice,
	     cudaGetDeviceCount,
	     cudaMalloc,
	     cudaFree,
	     cudaMemcpy,
	     cudaMemcpyKind,
	     cudaHostGetDevicePointer,
	     cudaGetErrorString,
	     cudaGetErrorName,
	     cudaHostRegister,
	     cudaHostRegisterDefault,
	     cudaHostUnregister,
	     cudaStreamCreate,
	     cudaStreamDestroy,
	     cudaEventCreateWithFlags,
	     cudaEventDestroy,
	     cudaEventDisableTiming,
	     cudaEventRecord,
	     cudaEventQuery],
					[
						CUDA_LIBS="-lcudart -lcuda -lnvrtc"
						ac_use_cuda=yes
					],
					[
						ac_use_cuda=no
					]
				)
			fi
		fi

		AM_CONDITIONAL([USE_CUDA], [test x"${ac_use_cuda}" = x"yes"])
		if test x"${ac_use_cuda}" = x"yes" ; then
			AC_DEFINE([USE_CUDA], [1], [Define if CUDA is enabled.])
		fi

		AM_CONDITIONAL([USE_CUDA_CL], [test x"${ac_use_cuda_cl}" = x"yes"])
		if test x"${ac_use_cuda_cl}" = x"yes" ; then
			AC_DEFINE([USE_CUDA_CL], [1], [Define if CUDA OpenCL is enabled.])
		fi

		AC_SUBST([CUDA_LIBS])
		AC_SUBST([CUDA_CFLAGS])
	]
)
