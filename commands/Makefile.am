AM_CXXFLAGS = $(PTHREAD_CFLAGS)
AM_LDFLAGS = 
CXX = $(NANOS6_MCXX)

if MCC_SUPORTS_EXTERNAL_INSTALL
AM_CXXFLAGS += --no-default-nanos6-inc --no-default-nanos6-lib -I$(top_srcdir)/api -I$(top_builddir) -include nanos6.h 
nanos6_info_LDADD = $(top_builddir)/libnanos6-main-wrapper.a ../libnanos6.la -ldl
endif


bin_PROGRAMS =

if HAVE_NANOS6_MERCURIUM
bin_PROGRAMS += nanos6-info
endif


nanos6_info_SOURCES = nanos6-info.cpp
nanos6_info_CPPFLAGS = -DNDEBUG
nanos6_info_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
