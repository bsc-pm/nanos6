#	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
#
#	Copyright (C) 2015-2021 Barcelona Supercomputing Center (BSC)

bin_PROGRAMS = nanos6-info nanos6-mergeprv cn6

nanos6_info_SOURCES = nanos6-info.cpp
nanos6_info_CPPFLAGS = -DNDEBUG
nanos6_info_CXXFLAGS = $(OPT_CXXFLAGS) $(PTHREAD_CFLAGS) -I$(top_srcdir)/api
nanos6_info_LDFLAGS = -Wl,-z,lazy $(jemalloc_LIBS)
nanos6_info_LDADD = $(top_builddir)/nanos6-library-mode.o ../libnanos6.la -ldl

nanos6_mergeprv_SOURCES = mergeprv.c

libprv_la_SOURCES = prv.c pcf.c
libprv_la_CPPFLAGS = $(babeltrace2_CPPFLAGS)

customlibdir = $(libdir)/nanos6
customlib_LTLIBRARIES = libprv.la

cn6_SOURCES = cn6.c
cn6_LDFLAGS = $(babeltrace2_LIBS)
cn6_CPPFLAGS = $(babeltrace2_CPPFLAGS) -DPRV_LIB_PATH='"$(customlibdir)"'
