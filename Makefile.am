ACLOCAL_AMFLAGS = -I m4
AM_CXXFLAGS = -I$(srcdir)/src


lib_LTLIBRARIES = libsimpless.la libsimpless-debug.la
noinst_LTLIBRARIES = libcheck-shutdown.la libcheck-shutdown-debug.la

# See info page of libtool "Updating version info"
lib_current = 0
lib_revision = 0
lib_age = 0

common_sources = \
	src/executors/threads/CPU.cpp \
	src/executors/threads/CPU.hpp \
	src/executors/threads/ThreadManager.cpp \
	src/executors/threads/ThreadManager.hpp \
	src/executors/threads/WorkerThread.cpp \
	src/executors/threads/WorkerThread.hpp \
	src/hardware/places/CPUPlace.hpp \
	src/hardware/places/HardwarePlace.hpp \
	src/lowlevel/apple/SpinLockImplementation.hpp \
	src/lowlevel/ConditionVariable.hpp \
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
	src/system/ompss/AddTask.hpp \
	src/system/ompss/TaskWait.cpp \
	src/system/ompss/TaskWait.hpp \
	src/tasks/Task.hpp


noinst_HEADERS = tests/infrastructure/TestAnyProtocolProducer.hpp
EXTRA_DIST = tests/infrastructure/tap-driver.sh


libsimpless_la_CPPFLAGS = -DNDEBUG
libsimpless_la_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
libsimpless_la_SOURCES = $(common_sources)
libsimpless_la_LDFLAGS = -version-info $(lib_current):$(lib_revision):$(lib_age) -lpthread -ldl

libsimpless_debug_la_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
libsimpless_debug_la_SOURCES = $(common_sources)
libsimpless_debug_la_LDFLAGS = -version-info $(lib_current):$(lib_revision):$(lib_age) -lpthread -ldl



#
# Tests
#

enabled_tests = \
	noop.debug.test noop.test \
	single-nest.debug.test single-nest.test \
	fibonacci.debug.test fibonacci.test

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


test_common_debug_libs = libsimpless-debug.la libcheck-shutdown-debug.la
test_common_libs = libsimpless.la libcheck-shutdown.la
test_common_ldflags = -no-install

noop_debug_test_SOURCES = tests/apiless/noop/noop.cpp
noop_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
noop_debug_test_LDADD = $(test_common_debug_libs)
noop_debug_test_LDFLAGS = $(test_common_ldflags)

noop_test_SOURCES = tests/apiless/noop/noop.cpp
noop_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
noop_test_LDADD = $(test_common_libs)
noop_test_LDFLAGS = $(test_common_ldflags)

single_nest_debug_test_SOURCES = tests/apiless/single-nest/single-nest.cpp
single_nest_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
single_nest_debug_test_LDADD = $(test_common_debug_libs)
single_nest_debug_test_LDFLAGS = $(test_common_ldflags)

single_nest_test_SOURCES = tests/apiless/single-nest/single-nest.cpp
single_nest_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
single_nest_test_LDADD = $(test_common_libs)
single_nest_test_LDFLAGS = $(test_common_ldflags)

fibonacci_debug_test_SOURCES = tests/apiless/fibonacci/fibonacci.cpp
fibonacci_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
fibonacci_debug_test_LDADD = $(test_common_debug_libs)
fibonacci_debug_test_LDFLAGS = $(test_common_ldflags)

fibonacci_test_SOURCES = tests/apiless/fibonacci/fibonacci.cpp
fibonacci_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
fibonacci_test_LDADD = $(test_common_libs)
fibonacci_test_LDFLAGS = $(test_common_ldflags)


check_PROGRAMS = $(enabled_tests) $(disabled_tests)
TESTS = $(enabled_tests)
TEST_LOG_DRIVER = env AM_TAP_AWK='$(AWK)' $(SHELL) $(top_srcdir)/tests/infrastructure/tap-driver.sh





libtool: $(LIBTOOL_DEPS)
	$(SHELL) ./config.status libtool


