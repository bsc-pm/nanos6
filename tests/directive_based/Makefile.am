AM_CXXFLAGS = -I$(top_srcdir)/api -I$(top_builddir) -I$(top_srcdir)/tests $(PTHREAD_CFLAGS)
AM_LDFLAGS = -L$(top_builddir)/.libs -rpath $(abs_top_builddir)/.libs
CXX = $(NANOS6_MCXX) --parallel


#
# Tests
#

base_tests =
cpu_management_tests =
user_mutex_tests =
linear_region_tests =


if HAVE_NANOS6_MERCURIUM
base_tests += \
	fibonacci.debug.test fibonacci.test \
	dep-nonest.debug.test dep-nonest.test \
	dep-early-release.debug.test dep-early-release.test \
	dep-er-and-weak.debug.test dep-er-and-weak.test \
	if0.debug.test if0.test

cpu_management_tests += \
	cpu-activation.debug.test cpu-activation.test

user_mutex_tests += \
	critical.debug.test critical.test

linear_region_tests += \
	lr-nonest.debug.test lr-nonest.test \
	lr-nonest-upgrades.debug.test lr-nonest-upgrades.test \
	lr-early-release.debug.test lr-early-release.test  \
	lr-er-and-weak.debug.test lr-er-and-weak.test
endif


check_PROGRAMS = $(base_tests)
TESTS = $(base_tests)


if HAVE_CPU_MANAGEMENT
check_PROGRAMS += $(cpu_management_tests)
TESTS += $(cpu_management_tests)
endif

if HAVE_WORKING_USER_MUTEX
check_PROGRAMS += $(user_mutex_tests)
TESTS += $(user_mutex_tests)
endif

if HAVE_LINEAR_REGION_SUPPORT
check_PROGRAMS += $(linear_region_tests)
TESTS += $(linear_region_tests)
endif

test_common_debug_ldflags = -no-install $(AM_LDFLAGS) $(PTHREAD_CFLAGS) $(PTHREAD_LIBS)
test_common_ldflags = -no-install $(AM_LDFLAGS) $(PTHREAD_CFLAGS) $(PTHREAD_LIBS)

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

dep_nonest_debug_test_SOURCES = dependencies/dep-nonest.cpp
dep_nonest_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
dep_nonest_debug_test_LDFLAGS = $(test_common_debug_ldflags)

dep_nonest_test_SOURCES = dependencies/dep-nonest.cpp
dep_nonest_test_CPPFLAGS = -DNDEBUG
dep_nonest_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
dep_nonest_test_LDFLAGS = $(test_common_ldflags)

dep_early_release_debug_test_SOURCES = dependencies/dep-early-release.cpp
dep_early_release_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
dep_early_release_debug_test_LDFLAGS = $(test_common_debug_ldflags)

dep_early_release_test_SOURCES = dependencies/dep-early-release.cpp
dep_early_release_test_CPPFLAGS = -DNDEBUG
dep_early_release_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
dep_early_release_test_LDFLAGS = $(test_common_ldflags)

dep_er_and_weak_debug_test_SOURCES = dependencies/dep-er-and-weak.cpp
dep_er_and_weak_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
dep_er_and_weak_debug_test_LDFLAGS = $(test_common_debug_ldflags)

dep_er_and_weak_test_SOURCES = dependencies/dep-er-and-weak.cpp
dep_er_and_weak_test_CPPFLAGS = -DNDEBUG
dep_er_and_weak_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
dep_er_and_weak_test_LDFLAGS = $(test_common_ldflags)

if0_debug_test_SOURCES = dependencies/if0.cpp
if0_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
if0_debug_test_LDFLAGS = $(test_common_debug_ldflags)

if0_test_SOURCES = dependencies/if0.cpp
if0_test_CPPFLAGS = -DNDEBUG
if0_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
if0_test_LDFLAGS = $(test_common_ldflags)

lr_nonest_debug_test_SOURCES = linear-regions/lr-nonest.cpp
lr_nonest_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
lr_nonest_debug_test_LDFLAGS = $(test_common_debug_ldflags)

lr_nonest_test_SOURCES = linear-regions/lr-nonest.cpp
lr_nonest_test_CPPFLAGS = -DNDEBUG
lr_nonest_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
lr_nonest_test_LDFLAGS = $(test_common_ldflags)

lr_nonest_upgrades_debug_test_SOURCES = linear-regions/lr-nonest-upgrades.cpp
lr_nonest_upgrades_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
lr_nonest_upgrades_debug_test_LDFLAGS = $(test_common_debug_ldflags)

lr_nonest_upgrades_test_SOURCES = linear-regions/lr-nonest-upgrades.cpp
lr_nonest_upgrades_test_CPPFLAGS = -DNDEBUG
lr_nonest_upgrades_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
lr_nonest_upgrades_test_LDFLAGS = $(test_common_ldflags)

lr_early_release_debug_test_SOURCES = linear-regions/lr-early-release.cpp
lr_early_release_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
lr_early_release_debug_test_LDFLAGS = $(test_common_debug_ldflags)

lr_early_release_test_SOURCES = linear-regions/lr-early-release.cpp
lr_early_release_test_CPPFLAGS = -DNDEBUG
lr_early_release_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
lr_early_release_test_LDFLAGS = $(test_common_ldflags)

lr_er_and_weak_debug_test_SOURCES = linear-regions/lr-er-and-weak.cpp
lr_er_and_weak_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
lr_er_and_weak_debug_test_LDFLAGS = $(test_common_debug_ldflags)

lr_er_and_weak_test_SOURCES = linear-regions/lr-er-and-weak.cpp
lr_er_and_weak_test_CPPFLAGS = -DNDEBUG
lr_er_and_weak_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
lr_er_and_weak_test_LDFLAGS = $(test_common_ldflags)

if AWK_IS_SANE
TEST_LOG_DRIVER = env AM_TAP_AWK='$(AWK)' LD_LIBRARY_PATH='$(top_builddir)/.libs:${LD_LIBRARY_PATH}' $(SHELL) $(top_srcdir)/tests/select-version.sh $(SHELL) $(top_srcdir)/tests/tap-driver.sh
else
TEST_LOG_DRIVER = env AM_TAP_AWK='$(AWK)' LD_LIBRARY_PATH='$(top_builddir)/.libs:${LD_LIBRARY_PATH}' $(SHELL) $(top_srcdir)/tests/select-version.sh $(PERL) $(top_srcdir)/tests/tap-driver.pl 
endif

