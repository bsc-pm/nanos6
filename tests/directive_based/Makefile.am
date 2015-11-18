ACLOCAL_AMFLAGS = -I ../../m4
AM_CXXFLAGS = -I$(srcdir)/..

#
# Tests
#

alltests = \
	noop.debug.test noop.test \
	single-nest.debug.test single-nest.test \
	fibonacci.debug.test fibonacci.test \
	cpu-activation.debug.test cpu-activation.test \
	critical.debug.test critical.test \
	dependencies-nonest.debug.test dependencies-nonest.test

check_PROGRAMS =
TESTS =
if TEST_WITH_MCC
check_PROGRAMS += $(alltests)
TESTS += $(alltests)
endif


shutdown_common_sources = \
	tests/infrastructure/ProgramLifecycle.cpp

test_common_debug_ldflags = -no-install --WL,-Xlinker,-rpath,-Xlinker,$(abs_top_builddir)/../../.libs --WL,-L../../.libs,-lcheck-shutdown-debug
test_common_ldflags = -no-install --WL,-Xlinker,-rpath,-Xlinker,$(abs_top_builddir)/../../.libs --WL,-L../../.libs,-lcheck-shutdown

noop_debug_test_SOURCES = noop/noop.cpp
noop_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
noop_debug_test_LDFLAGS = $(test_common_debug_ldflags)

noop_test_SOURCES = noop/noop.cpp
noop_test_CPPFLAGS = -DNDEBUG
noop_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
noop_test_LDFLAGS = $(test_common_ldflags)

single_nest_debug_test_SOURCES = single-nest/single-nest.cpp
single_nest_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
single_nest_debug_test_LDFLAGS = $(test_common_debug_ldflags)

single_nest_test_SOURCES = single-nest/single-nest.cpp
single_nest_test_CPPFLAGS = -DNDEBUG
single_nest_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
single_nest_test_LDFLAGS = $(test_common_ldflags)

fibonacci_debug_test_SOURCES = fibonacci/fibonacci.cpp
fibonacci_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
fibonacci_debug_test_LDFLAGS = $(test_common_debug_ldflags)

fibonacci_test_SOURCES = fibonacci/fibonacci.cpp
fibonacci_test_CPPFLAGS = -DNDEBUG
fibonacci_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
fibonacci_test_LDFLAGS = $(test_common_debug_ldflags)

cpu_activation_debug_test_SOURCES = cpu-activation/cpu-activation.cpp cpu-activation/ConditionVariable.hpp
cpu_activation_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
cpu_activation_debug_test_LDFLAGS = $(test_common_debug_ldflags)

cpu_activation_test_SOURCES = cpu-activation/cpu-activation.cpp cpu-activation/ConditionVariable.hpp
cpu_activation_test_CPPFLAGS = -DNDEBUG
cpu_activation_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
cpu_activation_test_LDFLAGS = $(test_common_ldflags)

critical_debug_test_SOURCES = critical/critical.cpp
critical_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
critical_debug_test_LDFLAGS = $(test_common_debug_ldflags)

critical_test_SOURCES = critical/critical.cpp
critical_test_CPPFLAGS = -DNDEBUG
critical_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
critical_test_LDFLAGS = $(test_common_ldflags)

dependencies_nonest_debug_test_SOURCES = dependencies-nonest/dependencies-nonest.cpp
dependencies_nonest_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
dependencies_nonest_debug_test_LDFLAGS = $(test_common_debug_ldflags)

dependencies_nonest_test_SOURCES = dependencies-nonest/dependencies-nonest.cpp
dependencies_nonest_test_CPPFLAGS = -DNDEBUG
dependencies_nonest_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
dependencies_nonest_test_LDFLAGS = $(test_common_ldflags)


if HAVE_NANOS6_LIBRARY_PATH
TEST_LOG_DRIVER = env AM_TAP_AWK='$(AWK)' NANOS6_LIBRARY_PATH='$(NANOS6_LIBRARY_PATH)' $(SHELL) $(top_srcdir)/../../tests/infrastructure/tap-driver.sh
else
TEST_LOG_DRIVER = env AM_TAP_AWK='$(AWK)' $(SHELL) $(top_srcdir)/../../tests/infrastructure/tap-driver.sh
endif



DISTCLEANFILES = $(LOCAL_MCC_CONFIG)


libtool: $(LIBTOOL_DEPS)
	$(SHELL) ./config.status libtool


