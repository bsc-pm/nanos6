#!/bin/sh

srcdir=$(dirname $0)
prefix=/usr/local
for option in $* ; do
	case "${option}" in
	--prefix=*)
		prefix=$(echo "${option}" | sed 's/--prefix=//g')
		;;
	esac
done

builddir=$(pwd)
top_builddir="${builddir}/../.."
nanos6_libdir="${top_builddir}/.libs"

exec "${srcdir}/configure" --with-nanos6-libdir="${nanos6_libdir}" "$@"
