ACLOCAL_AMFLAGS = -I common/m4
AM_CXXFLAGS = -I$(srcdir)/src -I$(builddir)/src -I$(top_srcdir)/common -I$(top_srcdir)/common/api

SUBDIRS = . tests/directive_based/check-shutdown tests/directive_based/tests


lib_LTLIBRARIES = \
	libnanos6-optimized.la libnanos6-debug.la \
	libnanos6-graph.la \
	libnanos6-profile.la \
	libnanos6-stats.la \
	libnanos6-null.la libnanos6-null-cb.la libnanos6-null-sb.la 

# See info page of libtool "Updating version info"
lib_current = 0
lib_revision = 0
lib_age = 0

pkginclude_HEADERS = \
	common/api/nanos6_rt_interface.h \
	common/api/nanos6_debug_interface.h

nodist_noinst_HEADERS = \
	src/version/VersionInfo.hpp


# Sources for debugging both the regular and the null runtime
universal_debug_sources = \
	src/system/debug/RuntimeVersionAPI.cpp


common_sources = \
	src/dependencies/DataAccess.hpp \
	src/dependencies/DataAccessRange.hpp \
	src/dependencies/DataAccessRegistration.hpp \
	src/dependencies/DataAccessSequenceLinkingArtifactsImplementation.hpp \
	src/dependencies/DataAccessSequence.hpp \
	src/dependencies/DataAccessSequenceImplementation.hpp \
	src/dependencies/DataAccessSequenceLinkingArtifacts.hpp \
	src/dependencies/DataAccessType.hpp \
	src/dependencies/FixedAddressDataAccessMap.hpp \
	src/dependencies/TaskDataAccessesLinkingArtifacts.hpp \
	src/dependencies/TaskDataAccessesLinkingArtifactsImplementation.hpp \
	src/executors/threads/CPU.cpp \
	src/executors/threads/CPU.hpp \
	src/executors/threads/CPUActivation.hpp \
	src/executors/threads/DefaultThreadManagerPolicy.cpp \
	src/executors/threads/DefaultThreadManagerPolicy.hpp \
	src/executors/threads/ThreadManager.cpp \
	src/executors/threads/ThreadManager.hpp \
	src/executors/threads/ThreadManagerPolicy.cpp \
	src/executors/threads/ThreadManagerPolicy.hpp \
	src/executors/threads/ThreadManagerPolicyInterface.hpp \
	src/executors/threads/WorkerThread.cpp \
	src/executors/threads/WorkerThread.hpp \
	src/hardware/places/CPUPlace.hpp \
	src/hardware/places/HardwarePlace.hpp \
	src/instrument/InstrumentAddTask.hpp \
	src/instrument/InstrumentDependenciesByAccess.hpp \
	src/instrument/InstrumentDependenciesByAccessSequences.hpp \
	src/instrument/InstrumentDependenciesByGroup.hpp \
	src/instrument/InstrumentInitAndShutdown.hpp \
	src/instrument/InstrumentTaskExecution.hpp \
	src/instrument/InstrumentTaskStatus.hpp \
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
	src/lowlevel/BoostAssertionFailureHandler.cpp \
	src/support/ConstPropagator.hpp \
	src/support/GlobalLock.cpp \
	src/support/GlobalLock.hpp \
	src/support/InlineDoublyLinkedList.hpp \
	src/system/Bootstrap.cpp \
	src/system/LeaderThread.cpp \
	src/system/LeaderThread.hpp \
	src/system/debug/DebugAPI.cpp \
	src/system/ompss/AddTask.cpp \
	src/system/ompss/RegisterDependencies.cpp \
	src/system/ompss/TaskBlocking.cpp \
	src/system/ompss/TaskBlocking.hpp \
	src/system/ompss/TaskWait.cpp \
	src/system/ompss/UserMutex.cpp \
	src/system/ompss/UserMutex.hpp \
	src/tasks/Task.hpp \
	src/tasks/TaskDebuggingInterface.hpp

noinstrument_sources = \
	src/instrument/null/InstrumentAddTask.hpp \
	src/instrument/null/InstrumentDataAccessId.hpp \
	src/instrument/null/InstrumentDataAccessSequenceId.hpp \
	src/instrument/null/InstrumentDependenciesByAccess.hpp \
	src/instrument/null/InstrumentDependenciesByAccessSequences.hpp \
	src/instrument/null/InstrumentDependenciesByGroup.hpp \
	src/instrument/null/InstrumentInitAndShutdown.hpp \
	src/instrument/null/InstrumentTaskExecution.hpp \
	src/instrument/null/InstrumentTaskId.hpp \
	src/instrument/null/InstrumentTaskStatus.hpp \
	src/instrument/null/InstrumentTaskWait.hpp \
	src/instrument/null/InstrumentThreadManagement.hpp \
	src/instrument/null/InstrumentUserMutex.hpp

instrument_graph_sources = \
	src/instrument/graph/Color.hpp \
	src/instrument/graph/InstrumentAddTask.cpp \
	src/instrument/graph/InstrumentAddTask.hpp \
	src/instrument/graph/InstrumentDataAccessId.hpp \
	src/instrument/graph/InstrumentDataAccessSequenceId.hpp \
	src/instrument/graph/InstrumentDependenciesByAccess.cpp \
	src/instrument/graph/InstrumentDependenciesByAccess.hpp \
	src/instrument/graph/InstrumentDependenciesByAccessSequences.cpp \
	src/instrument/graph/InstrumentDependenciesByAccessSequences.hpp \
	src/instrument/graph/InstrumentDependenciesByGroup.hpp \
	src/instrument/graph/InstrumentGraph.cpp \
	src/instrument/graph/InstrumentGraph.hpp \
	src/instrument/graph/InstrumentInitAndShutdown.cpp \
	src/instrument/graph/InstrumentInitAndShutdown.hpp \
	src/instrument/graph/InstrumentTaskExecution.cpp \
	src/instrument/graph/InstrumentTaskExecution.hpp \
	src/instrument/graph/InstrumentTaskId.hpp \
	src/instrument/graph/InstrumentTaskStatus.hpp \
	src/instrument/graph/InstrumentTaskWait.cpp \
	src/instrument/graph/InstrumentTaskWait.hpp \
	src/instrument/graph/InstrumentThreadManagement.cpp \
	src/instrument/graph/InstrumentThreadManagement.hpp \
	src/instrument/graph/InstrumentUserMutex.cpp \
	src/instrument/graph/InstrumentUserMutex.hpp

instrument_profile_sources = \
	src/instrument/profile/InstrumentAddTask.hpp \
	src/instrument/profile/InstrumentDataAccessId.hpp \
	src/instrument/profile/InstrumentDataAccessSequenceId.hpp \
	src/instrument/profile/InstrumentDependenciesByAccess.hpp \
	src/instrument/profile/InstrumentDependenciesByAccessSequences.hpp \
	src/instrument/profile/InstrumentDependenciesByGroup.hpp \
	src/instrument/profile/InstrumentInitAndShutdown.cpp \
	src/instrument/profile/InstrumentInitAndShutdown.hpp \
	src/instrument/profile/InstrumentProfile.cpp \
	src/instrument/profile/InstrumentProfile.hpp \
	src/instrument/profile/InstrumentTaskExecution.hpp \
	src/instrument/profile/InstrumentTaskId.hpp \
	src/instrument/profile/InstrumentTaskStatus.hpp \
	src/instrument/profile/InstrumentTaskWait.hpp \
	src/instrument/profile/InstrumentThreadManagement.hpp \
	src/instrument/profile/InstrumentUserMutex.hpp

instrument_stats_sources = \
	src/instrument/stats/InstrumentAddTask.hpp \
	src/instrument/stats/InstrumentDataAccessId.hpp \
	src/instrument/stats/InstrumentDataAccessSequenceId.hpp \
	src/instrument/stats/InstrumentDependenciesByAccess.hpp \
	src/instrument/stats/InstrumentDependenciesByAccessSequences.hpp \
	src/instrument/stats/InstrumentDependenciesByGroup.hpp \
	src/instrument/stats/InstrumentInitAndShutdown.cpp \
	src/instrument/stats/InstrumentInitAndShutdown.hpp \
	src/instrument/stats/InstrumentStats.cpp \
	src/instrument/stats/InstrumentStats.hpp \
	src/instrument/stats/InstrumentTaskExecution.hpp \
	src/instrument/stats/InstrumentTaskId.hpp \
	src/instrument/stats/InstrumentTaskStatus.hpp \
	src/instrument/stats/InstrumentTaskWait.hpp \
	src/instrument/stats/InstrumentThreadManagement.hpp \
	src/instrument/stats/InstrumentUserMutex.hpp \
	src/instrument/stats/Timer.hpp

null_common_sources = \
	src/null/NullDebug.cpp \
	src/null/NullAPI.cpp

null_sources = \
	$(null_common_sources) \
	src/null/NullDynamicBlock.cpp

null_cb_sources = \
	$(null_common_sources) \
	src/null/NullConstantBlock.cpp

null_sb_sources = \
	$(null_common_sources) \
	src/null/NullStaticBlock.cpp

noinst_HEADERS = \
	common/tests/TestAnyProtocolProducer.hpp \
	common/tests/Timer.hpp

EXTRA_DIST = \
	common/tests/tap-driver.sh


common_libnanos6_cppflags = $(BOOST_CPPFLAGS) -DBOOST_ENABLE_ASSERT_DEBUG_HANDLER
common_libnanos6_ldflags = $(BOOST_LDFLAGS) -version-info $(lib_current):$(lib_revision):$(lib_age) -lpthread -ldl

libnanos6_optimized_la_CPPFLAGS = -DNDEBUG $(common_libnanos6_cppflags) -I$(srcdir)/src/instrument/null
libnanos6_optimized_la_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
libnanos6_optimized_la_SOURCES = $(common_sources) $(noinstrument_sources) $(universal_debug_sources)
libnanos6_optimized_la_LDFLAGS = $(common_libnanos6_ldflags)

libnanos6_debug_la_CPPFLAGS = $(common_libnanos6_cppflags) -I$(srcdir)/src/instrument/null
libnanos6_debug_la_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
libnanos6_debug_la_SOURCES = $(common_sources) $(noinstrument_sources) $(universal_debug_sources)
libnanos6_debug_la_LDFLAGS = $(common_libnanos6_ldflags)

libnanos6_graph_la_CPPFLAGS = -DNDEBUG $(common_libnanos6_cppflags) -I$(srcdir)/src/instrument/graph
libnanos6_graph_la_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
libnanos6_graph_la_SOURCES = $(common_sources) $(instrument_graph_sources) $(universal_debug_sources)
libnanos6_graph_la_LDFLAGS = $(common_libnanos6_ldflags)

libnanos6_profile_la_CPPFLAGS = -DNDEBUG $(common_libnanos6_cppflags) -I$(srcdir)/src/instrument/profile
libnanos6_profile_la_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS) -g2
libnanos6_profile_la_SOURCES = $(common_sources) $(instrument_profile_sources) $(universal_debug_sources)
libnanos6_profile_la_LDFLAGS = $(common_libnanos6_ldflags) $(BACKTRACE_LIBS)

libnanos6_stats_la_CPPFLAGS = -DNDEBUG $(common_libnanos6_cppflags) -I$(srcdir)/src/instrument/stats
libnanos6_stats_la_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS) -g2
libnanos6_stats_la_SOURCES = $(common_sources) $(instrument_stats_sources) $(universal_debug_sources)
libnanos6_stats_la_LDFLAGS = $(common_libnanos6_ldflags)

libnanos6_null_la_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
libnanos6_null_la_SOURCES = $(null_sources) $(universal_debug_sources)
libnanos6_null_la_LDFLAGS = $(common_libnanos6_ldflags)

libnanos6_null_cb_la_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
libnanos6_null_cb_la_SOURCES = $(null_cb_sources) $(universal_debug_sources)
libnanos6_null_cb_la_LDFLAGS = $(common_libnanos6_ldflags)

libnanos6_null_sb_la_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
libnanos6_null_sb_la_SOURCES = $(null_sb_sources) $(universal_debug_sources)
libnanos6_null_sb_la_LDFLAGS = $(common_libnanos6_ldflags)



#
# Tests
#

unit_tests = inline-double-linked-list.debug.test inline-double-linked-list.test

unit_test_common_cxxflags = -I$(top_srcdir)/common/tests

inline_double_linked_list_debug_test_SOURCES = tests/unit/support/TestInlineDoublyLinkedList.cpp
inline_double_linked_list_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS) $(unit_test_common_cxxflags)

inline_double_linked_list_test_SOURCES = tests/unit/support/TestInlineDoublyLinkedList.cpp
inline_double_linked_list_test_CPPFLAGS = -DNDEBUG
inline_double_linked_list_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS) $(unit_test_common_cxxflags)



check_PROGRAMS = $(unit_tests)
TESTS = $(unit_tests)
TEST_LOG_DRIVER = env AM_TAP_AWK='$(AWK)' $(SHELL) $(top_srcdir)/common/tests/tap-driver.sh





libtool: $(LIBTOOL_DEPS)
	$(SHELL) ./config.status libtool


src/version/VersionInfo.hpp: $(srcdir)/src/version/VersionInfo.hpp.in
	$(SHELL) ./config.status src/version/VersionInfo.hpp



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

