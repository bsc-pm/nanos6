#	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
#
#	Copyright (C) 2015-2021 Barcelona Supercomputing Center (BSC)

bin_PROGRAMS = nanos6-info

if BUILD_CTF2PRV_FAST
bin_PROGRAMS += nanos6-mergeprv nanos6-ctf2prv-fast
auxiliarylib_LTLIBRARIES = libprv.la
endif

nanos6_info_SOURCES = nanos6-info.cpp
nanos6_info_CPPFLAGS = -DNDEBUG
nanos6_info_CXXFLAGS = $(OPT_CXXFLAGS) $(PTHREAD_CFLAGS) -I$(top_srcdir)/api
nanos6_info_LDFLAGS = -Wl,-z,lazy $(jemalloc_LIBS)
nanos6_info_LDADD = $(top_builddir)/nanos6-library-mode.o ../libnanos6.la -ldl

nanos6_mergeprv_SOURCES = nanos6-mergeprv.c

libprv_la_SOURCES = libprv/pcf.c libprv/prv.c
libprv_la_CPPFLAGS = $(babeltrace2_CPPFLAGS)

auxiliarylibdir = $(libdir)/nanos6

nanos6_ctf2prv_fast_SOURCES = nanos6-ctf2prv-fast.c
nanos6_ctf2prv_fast_LDFLAGS = $(babeltrace2_LIBS)
nanos6_ctf2prv_fast_CPPFLAGS = $(babeltrace2_CPPFLAGS) -DPRV_LIB_PATH='"$(auxiliarylibdir)"'

noinst_HEADERS = \
	libprv/hwc.h \
	libprv/pcf.h \
	libprv/prv.h \
	libprv/uthash.h
