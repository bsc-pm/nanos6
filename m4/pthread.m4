AC_DEFUN([CHECK_PTHREAD],
	[
		AC_REQUIRE([AC_CANONICAL_HOST])
		AC_REQUIRE([AX_PTHREAD])
		
		# AX_PTHREAD does not seem to interact well with libtool on "regular" linux
		case $host_os in
			*linux*android*)
				;;
			*linux*)
				PTHREAD_LIBS="${PTHREAD_LIBS} -lpthread"
				;;
		esac
	]
)

