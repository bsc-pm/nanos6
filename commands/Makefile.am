#	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
#	
#	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)


AM_CXXFLAGS = $(PTHREAD_CFLAGS) -I$(top_srcdir)/api
AM_LDFLAGS = 
nanos6_info_LDADD = $(top_builddir)/nanos6-library-mode.o ../libnanos6.la -ldl

bin_PROGRAMS = nanos6-info

nanos6_info_SOURCES = nanos6-info.cpp
nanos6_info_CPPFLAGS = -DNDEBUG
nanos6_info_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
