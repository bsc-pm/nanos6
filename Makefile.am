ACLOCAL_AMFLAGS = -I m4
AM_CXXFLAGS = -I$(srcdir)/src

SUBDIRS = tests/directive_based


lib_LTLIBRARIES = libnanos6.la libnanos6-debug.la libnanos6-graph.la libnanos6-null.la
noinst_LTLIBRARIES = libcheck-shutdown.la libcheck-shutdown-debug.la

# See info page of libtool "Updating version info"
lib_current = 0
lib_revision = 0
lib_age = 0

pkginclude_HEADERS = \
	src/api/nanos6_rt_interface.h


common_sources = \
	src/executors/threads/CPU.cpp \
	src/executors/threads/CPU.hpp \
	src/executors/threads/CPUActivation.hpp \
	src/executors/threads/DefaultThreadManagerPolicy.cpp \
	src/executors/threads/DefaultThreadManagerPolicy.hpp \
	src/executors/threads/ThreadManager.cpp \
	src/executors/threads/ThreadManager.hpp \
	src/executors/threads/ThreadManagerDebuggingInterface.hpp \
	src/executors/threads/ThreadManagerPolicy.cpp \
	src/executors/threads/ThreadManagerPolicy.hpp \
	src/executors/threads/ThreadManagerPolicyInterface.hpp \
	src/executors/threads/WorkerThread.cpp \
	src/executors/threads/WorkerThread.hpp \
	src/hardware/places/CPUPlace.hpp \
	src/hardware/places/HardwarePlace.hpp \
	src/instrument/InstrumentAddTask.hpp \
	src/instrument/InstrumentInitAndShutdown.hpp \
	src/instrument/InstrumentTaskExecution.hpp \
	src/instrument/InstrumentTaskWait.hpp \
	src/instrument/InstrumentThreadManagement.hpp \
	src/instrument/InstrumentUserMutex.hpp \
	src/lowlevel/apple/SpinLockImplementation.hpp \
	src/lowlevel/ConditionVariable.hpp \
	src/lowlevel/EnvironmentVariable.hpp \
	src/lowlevel/FatalErrorHandler.hpp \
	src/lowlevel/posix/SpinLockImplementation.hpp \
	src/lowlevel/SpinLock.hpp \
	src/scheduling/DefaultScheduler.hpp \
	src/scheduling/NaiveScheduler.cpp \
	src/scheduling/NaiveScheduler.hpp \
	src/scheduling/Scheduler.cpp \
	src/scheduling/Scheduler.hpp \
	src/scheduling/SchedulerInterface.hpp \
	src/system/Bootstrap.cpp \
	src/system/LeaderThread.cpp \
	src/system/LeaderThread.hpp \
	src/system/MainTask.cpp \
	src/system/MainTask.hpp \
	src/system/ompss/AddTask.cpp \
	src/system/ompss/TaskBlocking.cpp \
	src/system/ompss/TaskBlocking.hpp \
	src/system/ompss/TaskWait.cpp \
	src/system/ompss/UserMutex.cpp \
	src/system/ompss/UserMutex.hpp \
	src/tasks/Task.hpp \
	src/tasks/TaskDebuggingInterface.hpp

noinstrument_sources = \
	src/instrument/null/InstrumentAddTask.hpp \
	src/instrument/null/InstrumentInitAndShutdown.hpp \
	src/instrument/null/InstrumentTaskExecution.hpp \
	src/instrument/null/InstrumentTaskId.hpp \
	src/instrument/null/InstrumentTaskWait.hpp \
	src/instrument/null/InstrumentThreadManagement.hpp \
	src/instrument/null/InstrumentUserMutex.hpp

instrument_graph_sources = \
	src/instrument/graph/Color.hpp \
	src/instrument/graph/InstrumentAddTask.cpp \
	src/instrument/graph/InstrumentAddTask.hpp \
	src/instrument/graph/InstrumentGraph.cpp \
	src/instrument/graph/InstrumentGraph.hpp \
	src/instrument/graph/InstrumentInitAndShutdown.cpp \
	src/instrument/graph/InstrumentInitAndShutdown.hpp \
	src/instrument/graph/InstrumentTaskExecution.cpp \
	src/instrument/graph/InstrumentTaskExecution.hpp \
	src/instrument/graph/InstrumentTaskId.hpp \
	src/instrument/graph/InstrumentTaskWait.cpp \
	src/instrument/graph/InstrumentTaskWait.hpp \
	src/instrument/graph/InstrumentThreadManagement.cpp \
	src/instrument/graph/InstrumentThreadManagement.hpp \
	src/instrument/graph/InstrumentUserMutex.cpp \
	src/instrument/graph/InstrumentUserMutex.hpp

null_sources = \
	src/null/NullAPI.cpp

noinst_HEADERS = \
	tests/infrastructure/ProgramLifecycle.hpp \
	tests/infrastructure/TestAnyProtocolProducer.hpp \
	tests/infrastructure/Timer.hpp

EXTRA_DIST = \
	tests/infrastructure/tap-driver.sh


libnanos6_la_CPPFLAGS = -DNDEBUG -I$(srcdir)/src/instrument/null
libnanos6_la_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
libnanos6_la_SOURCES = $(common_sources) $(noinstrument_sources)
libnanos6_la_LDFLAGS = -version-info $(lib_current):$(lib_revision):$(lib_age) -lpthread -ldl

libnanos6_debug_la_CPPFLAGS = -I$(srcdir)/src/instrument/null
libnanos6_debug_la_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
libnanos6_debug_la_SOURCES = $(common_sources) $(noinstrument_sources)
libnanos6_debug_la_LDFLAGS = -version-info $(lib_current):$(lib_revision):$(lib_age) -lpthread -ldl

libnanos6_graph_la_CPPFLAGS = -DNDEBUG -I$(srcdir)/src/instrument/graph
libnanos6_graph_la_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
libnanos6_graph_la_SOURCES = $(common_sources) $(instrument_graph_sources)
libnanos6_graph_la_LDFLAGS = -version-info $(lib_current):$(lib_revision):$(lib_age) -lpthread -ldl

libnanos6_null_la_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
libnanos6_null_la_SOURCES = $(null_sources)
libnanos6_null_la_LDFLAGS = -version-info $(lib_current):$(lib_revision):$(lib_age) -lpthread -ldl



#
# Tests
#

enabled_tests = \
	noop.debug.test noop.test \
	single-nest.debug.test single-nest.test \
	fibonacci.debug.test fibonacci.test \
	cpu-activation.debug.test cpu-activation.test \
	critical.debug.test critical.test

disabled_tests =


shutdown_common_sources = \
	tests/infrastructure/ProgramLifecycle.cpp

libcheck_shutdown_la_CPPFLAGS = -DNDEBUG
libcheck_shutdown_la_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
libcheck_shutdown_la_SOURCES = $(shutdown_common_sources)
libcheck_shutdown_la_LDFLAGS = -module -rpath /nowhere -ldl

libcheck_shutdown_debug_la_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
libcheck_shutdown_debug_la_SOURCES = $(shutdown_common_sources)
libcheck_shutdown_debug_la_LDFLAGS = -module -rpath /nowhere -ldl


test_common_debug_libs = libnanos6-debug.la libcheck-shutdown-debug.la
test_common_libs = libnanos6.la libcheck-shutdown.la
test_common_ldflags = -no-install

noop_debug_test_SOURCES = tests/direct_api/noop/noop.cpp
noop_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
noop_debug_test_LDADD = $(test_common_debug_libs)
noop_debug_test_LDFLAGS = $(test_common_ldflags)

noop_test_SOURCES = tests/direct_api/noop/noop.cpp
noop_test_CPPFLAGS = -DNDEBUG
noop_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
noop_test_LDADD = $(test_common_libs)
noop_test_LDFLAGS = $(test_common_ldflags)

single_nest_debug_test_SOURCES = tests/direct_api/single-nest/single-nest.cpp
single_nest_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
single_nest_debug_test_LDADD = $(test_common_debug_libs)
single_nest_debug_test_LDFLAGS = $(test_common_ldflags)

single_nest_test_SOURCES = tests/direct_api/single-nest/single-nest.cpp
single_nest_test_CPPFLAGS = -DNDEBUG
single_nest_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
single_nest_test_LDADD = $(test_common_libs)
single_nest_test_LDFLAGS = $(test_common_ldflags)

fibonacci_debug_test_SOURCES = tests/direct_api/fibonacci/fibonacci.cpp
fibonacci_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
fibonacci_debug_test_LDADD = $(test_common_debug_libs)
fibonacci_debug_test_LDFLAGS = $(test_common_ldflags)

fibonacci_test_SOURCES = tests/direct_api/fibonacci/fibonacci.cpp
fibonacci_test_CPPFLAGS = -DNDEBUG
fibonacci_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
fibonacci_test_LDADD = $(test_common_libs)
fibonacci_test_LDFLAGS = $(test_common_ldflags)

cpu_activation_debug_test_SOURCES = tests/direct_api/cpu-activation/cpu-activation.cpp
cpu_activation_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS) -I$(srcdir)/src/instrument/null
cpu_activation_debug_test_LDADD = $(test_common_debug_libs) -lpthread
cpu_activation_debug_test_LDFLAGS = $(test_common_ldflags)

cpu_activation_test_SOURCES = tests/direct_api/cpu-activation/cpu-activation.cpp
cpu_activation_test_CPPFLAGS = -DNDEBUG
cpu_activation_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS) -I$(srcdir)/src/instrument/null
cpu_activation_test_LDADD = $(test_common_libs) -lpthread
cpu_activation_test_LDFLAGS = $(test_common_ldflags)

critical_debug_test_SOURCES = tests/direct_api/critical/critical.cpp
critical_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS) -I$(srcdir)/src/instrument/null
critical_debug_test_LDADD = $(test_common_debug_libs)
critical_debug_test_LDFLAGS = $(test_common_ldflags)

critical_test_SOURCES = tests/direct_api/critical/critical.cpp
critical_test_CPPFLAGS = -DNDEBUG
critical_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS) -I$(srcdir)/src/instrument/null
critical_test_LDADD = $(test_common_libs)
critical_test_LDFLAGS = $(test_common_ldflags)


check_PROGRAMS = $(enabled_tests) $(disabled_tests)
TESTS = $(enabled_tests)
TEST_LOG_DRIVER = env AM_TAP_AWK='$(AWK)' $(SHELL) $(top_srcdir)/tests/infrastructure/tap-driver.sh





libtool: $(LIBTOOL_DEPS)
	$(SHELL) ./config.status libtool




## --------------------------------- ##
## Format-independent Doxygen rules. ##
## --------------------------------- ##
if DX_COND_doc
## ------------------------------- ##
## Rules specific for HTML output. ##
## ------------------------------- ##
if DX_COND_html
DX_CLEAN_HTML = @DX_DOCDIR@/html
endif DX_COND_html
## ------------------------------ ##
## Rules specific for CHM output. ##
## ------------------------------ ##
if DX_COND_chm
DX_CLEAN_CHM = @DX_DOCDIR@/chm
if DX_COND_chi
DX_CLEAN_CHI = @DX_DOCDIR@/@PACKAGE@.chi
endif DX_COND_chi
endif DX_COND_chm
## ------------------------------ ##
## Rules specific for MAN output. ##
## ------------------------------ ##
if DX_COND_man
DX_CLEAN_MAN = @DX_DOCDIR@/man
endif DX_COND_man
## ------------------------------ ##
## Rules specific for RTF output. ##
## ------------------------------ ##
if DX_COND_rtf
DX_CLEAN_RTF = @DX_DOCDIR@/rtf
endif DX_COND_rtf
## ------------------------------ ##
## Rules specific for XML output. ##
## ------------------------------ ##
if DX_COND_xml
DX_CLEAN_XML = @DX_DOCDIR@/xml
endif DX_COND_xml
## ----------------------------- ##
## Rules specific for PS output. ##
## ----------------------------- ##
if DX_COND_ps
DX_CLEAN_PS = @DX_DOCDIR@/@PACKAGE@.ps
DX_PS_GOAL = doxygen-ps
doxygen-ps: @DX_DOCDIR@/@PACKAGE@.ps
@DX_DOCDIR@/@PACKAGE@.ps: @DX_DOCDIR@/@PACKAGE@.tag
	cd @DX_DOCDIR@/latex; \
	rm -f *.aux *.toc *.idx *.ind *.ilg *.log *.out; \
	$(DX_LATEX) refman.tex; \
	$(MAKEINDEX_PATH) refman.idx; \
	$(DX_LATEX) refman.tex; \
	countdown=5; \
	while $(DX_EGREP) 'Rerun (LaTeX|to get cross-references right)' \
		refman.log > /dev/null 2>&1 \
	&& test $$countdown -gt 0; do \
		$(DX_LATEX) refman.tex; \
		countdown=`expr $$countdown - 1`; \
	done; \
	$(DX_DVIPS) -o ../@PACKAGE@.ps refman.dvi
endif DX_COND_ps
## ------------------------------ ##
## Rules specific for PDF output. ##
## ------------------------------ ##
if DX_COND_pdf
DX_CLEAN_PDF = @DX_DOCDIR@/@PACKAGE@.pdf
DX_PDF_GOAL = doxygen-pdf
doxygen-pdf: @DX_DOCDIR@/@PACKAGE@.pdf
@DX_DOCDIR@/@PACKAGE@.pdf: @DX_DOCDIR@/@PACKAGE@.tag
	cd @DX_DOCDIR@/latex; \
	rm -f *.aux *.toc *.idx *.ind *.ilg *.log *.out; \
	$(DX_PDFLATEX) refman.tex; \
	$(DX_MAKEINDEX) refman.idx; \
	$(DX_PDFLATEX) refman.tex; \
	countdown=5; \
	while $(DX_EGREP) 'Rerun (LaTeX|to get cross-references right)' \
		refman.log > /dev/null 2>&1 \
	&& test $$countdown -gt 0; do \
		$(DX_PDFLATEX) refman.tex; \
		countdown=`expr $$countdown - 1`; \
	done; \
	mv refman.pdf ../@PACKAGE@.pdf
endif DX_COND_pdf
## ------------------------------------------------- ##
## Rules specific for LaTeX (shared for PS and PDF). ##
## ------------------------------------------------- ##
if DX_COND_latex
DX_CLEAN_LATEX = @DX_DOCDIR@/latex
endif DX_COND_latex
.PHONY: doxygen-run doxygen-doc $(DX_PS_GOAL) $(DX_PDF_GOAL)
.INTERMEDIATE: doxygen-run $(DX_PS_GOAL) $(DX_PDF_GOAL)
doxygen-run: @DX_DOCDIR@/@PACKAGE@.tag
doxygen-doc: doxygen-run $(DX_PS_GOAL) $(DX_PDF_GOAL)
@DX_DOCDIR@/@PACKAGE@.tag: $(DX_CONFIG) $(pkginclude_HEADERS)
	rm -rf @DX_DOCDIR@
	$(DX_ENV) $(DX_DOXYGEN) $(DX_CONFIG)
	echo Timestamp >$@
DX_CLEANFILES = \
	@DX_DOCDIR@/@PACKAGE@.tag \
	-r \
	$(DX_CLEAN_HTML) \
	$(DX_CLEAN_CHM) \
	$(DX_CLEAN_CHI) \
	$(DX_CLEAN_MAN) \
	$(DX_CLEAN_RTF) \
	$(DX_CLEAN_XML) \
	$(DX_CLEAN_PS) \
	$(DX_CLEAN_PDF) \
	$(DX_CLEAN_LATEX)
endif DX_COND_doc


MOSTLYCLEANFILES = DX_CLEANFILES

