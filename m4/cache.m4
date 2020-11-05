#	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
#
#	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)

AC_DEFUN([AC_CHECK_CACHE],
	[
		AC_ARG_VAR(
			[CACHELINE_WIDTH],
			[Specify the cacheline width of the target machine. By default, a best guess will be done on the current host.]
		)

		cache_size=""

		if test x"${CACHELINE_WIDTH}" != x"" ; then
			cache_size="${CACHELINE_WIDTH}"
		else
			AC_CHECK_FILE([/sys/devices/system/cpu/cpu0/cache/index3/coherency_line_size], [
				cache_size=$(</sys/devices/system/cpu/cpu0/cache/index3/coherency_line_size)

				if [ test -z "$cache_size" ] || [ test "$cache_size" -lt 1 ] ; then
					cache_size=0
				fi
			],[
				cache_size=0
			])

			if [ test -z "$cache_size" ] ; then
				AC_CHECK_FILE([/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size], [
					cache_size=$(</sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size)

					if [ test -z "$cache_size" ] || [ test "$cache_size" -lt 1 ] ; then
						AC_MSG_WARN([No cacheline found. Please specify with CACHELINE_WIDTH.])
						AC_MSG_WARN([Falling back to safe default.])
						cache_size=128
					fi
				],[
					AC_MSG_WARN([No cacheline found. Please specify with CACHELINE_WIDTH.])
					AC_MSG_WARN([Falling back to safe default.])
					cache_size=128
				])
			fi
		fi

		AC_MSG_CHECKING([the host cache line size])
		AC_MSG_RESULT([${cache_size}])

		AC_DEFINE_UNQUOTED([CACHELINE_SIZE], [${cache_size}], [Define cacheline width.])
	]
)
