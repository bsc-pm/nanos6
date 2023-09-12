#	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
#
#	Copyright (C) 2015-2023 Barcelona Supercomputing Center (BSC)

AM_CXXFLAGS = -I$(top_srcdir)/tests $(PTHREAD_CFLAGS)
AM_LDFLAGS =

CXX = $(NANOS6_MCXX) --parallel

if MCC_SUPORTS_EXTERNAL_INSTALL
AM_CXXFLAGS += --no-default-nanos6-inc --no-default-nanos6-lib -I$(top_srcdir)/api -I$(top_builddir) -include nanos6.h
AM_LDFLAGS += -L$(top_builddir)/.libs -rpath $(abs_top_builddir)/.libs -Wl,-z,lazy $(jemalloc_build_LIBS)
LDADD = $(top_builddir)/nanos6-main-wrapper.o $(top_builddir)/libnanos6.la -ldl
endif


if LESS_TEST_THREADS
AM_CXXFLAGS += -DTEST_LESS_THREADS
endif


#
# Tests
#

base_tests =
cpu_management_tests =
user_mutex_tests =
linear_region_tests =
reductions_tests =
discrete_tests =
dlb_tests =
numa_tests =


if HAVE_NANOS6_MERCURIUM
base_tests += \
	blocking.mercurium.test \
	events.mercurium.test \
	events-dep.mercurium.test \
	onready.mercurium.test \
	onready-events.mercurium.test \
	scheduling-wait-for.mercurium.test \
	fibonacci.mercurium.test \
	dep-nonest.mercurium.test \
	dep-early-release.mercurium.test \
	dep-er-and-weak.mercurium.test \
	if0.mercurium.test \
	dep-wait.mercurium.test \
	simple-commutative.mercurium.test \
	commutative-stencil.mercurium.test \
	taskloop-multiaxpy.mercurium.test \
	taskloop-dep-multiaxpy.mercurium.test \
	taskloop-nested-dep-multiaxpy.mercurium.test \
	taskloop-nonpod.mercurium.test \
	taskloop-nqueens.mercurium.test \
	taskloop-wait.mercurium.test

# Ignore CPU Activation test if we have DLB
# NOTE: The order of this tests should never change, new DLB-related
#       tests must be added under these
if HAVE_DLB
dlb_tests += \
	dlb-cpu-management.mercurium.test \
	dlb-cpu-sharing-active-process.mercurium.test \
	dlb-cpu-sharing-passive-process.mercurium.test \
	dlb-cpu-sharing.mercurium.test
else
cpu_management_tests += \
	cpu-activation.mercurium.test
endif

user_mutex_tests += \
	critical.mercurium.test

linear_region_tests += \
	lr-nonest.mercurium.test \
	lr-nonest-upgrades.mercurium.test \
	lr-early-release.mercurium.test  \
	lr-er-and-weak.mercurium.test \
	lr-release.mercurium.test

reductions_tests += \
	red-firstprivate.mercurium.test \
	red-nonest.mercurium.test \
	red-nest.mercurium.test \
	red-nest-other.mercurium.test \
	red-nqueens.mercurium.test \
	red-stress.mercurium.test

discrete_tests += \
	discrete-deps.mercurium.test \
	discrete-deps-nonest.mercurium.test \
	discrete-deps-nonest.mercurium.test \
	discrete-deps-early-release.mercurium.test \
	discrete-deps-er-and-weak.mercurium.test \
	discrete-deps-wait.mercurium.test \
	discrete-release.mercurium.test \
	discrete-simple-commutative.mercurium.test \
	discrete-red-stress.mercurium.test \
	discrete-taskloop-multiaxpy.mercurium.test \
	discrete-taskloop-dep-multiaxpy.mercurium.test \
	discrete-taskloop-nested-dep-multiaxpy.mercurium.test \
	discrete-taskloop-nonpod.mercurium.test \
	discrete-taskloop-nqueens.mercurium.test

numa_tests += \
	numa-allocations.mercurium.test \
	numa-auto.mercurium.test \
	numa-bitmask-manip.mercurium.test \
	numa-irregular-allocations.mercurium.test \
	numa-off.mercurium.test \
	numa-on.mercurium.test \
	numa-wildcards.mercurium.test

base_tests +=  \
	blocking.mercurium.debug.test \
	events.mercurium.debug.test \
	events-dep.mercurium.debug.test \
	onready.mercurium.debug.test \
	onready-events.mercurium.debug.test \
	scheduling-wait-for.mercurium.debug.test \
	fibonacci.mercurium.debug.test \
	dep-nonest.mercurium.debug.test \
	dep-early-release.mercurium.debug.test \
	dep-er-and-weak.mercurium.debug.test \
	if0.mercurium.debug.test \
	dep-wait.mercurium.debug.test \
	simple-commutative.mercurium.debug.test \
	commutative-stencil.mercurium.debug.test \
	taskloop-multiaxpy.mercurium.debug.test \
	taskloop-dep-multiaxpy.mercurium.debug.test \
	taskloop-nested-dep-multiaxpy.mercurium.debug.test \
	taskloop-nonpod.mercurium.debug.test \
	taskloop-nqueens.mercurium.debug.test \
	taskloop-wait.mercurium.debug.test

# Ignore CPU Activation test if we have DLB for now
if HAVE_DLB
dlb_tests += \
	dlb-cpu-management.mercurium.debug.test \
	dlb-cpu-sharing-active-process.mercurium.debug.test \
	dlb-cpu-sharing-passive-process.mercurium.debug.test \
	dlb-cpu-sharing.mercurium.debug.test
else
cpu_management_tests += \
	cpu-activation.mercurium.debug.test
endif

user_mutex_tests += \
	critical.mercurium.debug.test

linear_region_tests += \
	lr-nonest.mercurium.debug.test \
	lr-nonest-upgrades.mercurium.debug.test \
	lr-early-release.mercurium.debug.test  \
	lr-er-and-weak.mercurium.debug.test \
	lr-release.mercurium.debug.test

reductions_tests += \
	red-firstprivate.mercurium.debug.test \
	red-nonest.mercurium.debug.test \
	red-nest.mercurium.debug.test \
	red-nest-other.mercurium.debug.test \
	red-nqueens.mercurium.debug.test \
	red-stress.mercurium.debug.test

discrete_tests += \
	discrete-deps.mercurium.debug.test \
	discrete-deps-nonest.mercurium.debug.test \
	discrete-deps-nonest.mercurium.debug.test \
	discrete-deps-early-release.mercurium.debug.test \
	discrete-deps-er-and-weak.mercurium.debug.test \
	discrete-deps-wait.mercurium.debug.test \
	discrete-release.mercurium.debug.test \
	discrete-simple-commutative.mercurium.debug.test \
	discrete-red-stress.mercurium.debug.test \
	discrete-taskloop-multiaxpy.mercurium.debug.test \
	discrete-taskloop-dep-multiaxpy.mercurium.debug.test \
	discrete-taskloop-nested-dep-multiaxpy.mercurium.debug.test \
	discrete-taskloop-nonpod.mercurium.debug.test \
	discrete-taskloop-nqueens.mercurium.debug.test

numa_tests += \
	numa-allocations.mercurium.debug.test \
	numa-auto.mercurium.debug.test \
	numa-bitmask-manip.mercurium.debug.test \
	numa-irregular-allocations.mercurium.debug.test \
	numa-off.mercurium.debug.test \
	numa-on.mercurium.debug.test \
	numa-wildcards.mercurium.debug.test

endif


check_PROGRAMS = $(base_tests)
TESTS = $(base_tests)

check_PROGRAMS += $(cpu_management_tests)
TESTS += $(cpu_management_tests)

check_PROGRAMS += $(user_mutex_tests)
TESTS += $(user_mutex_tests)

if HAVE_LINEAR_REGION_SUPPORT
check_PROGRAMS += $(linear_region_tests)
TESTS += $(linear_region_tests)
endif

if HAVE_REDUCTIONS_SUPPORT
check_PROGRAMS += $(reductions_tests)
TESTS += $(reductions_tests)
endif

if DISCRETE_DEPENDENCIES
check_PROGRAMS += $(discrete_tests)
TESTS += $(discrete_tests)
check_PROGRAMS += $(numa_tests)
TESTS += $(numa_tests)
endif

if HAVE_DLB
check_PROGRAMS += $(dlb_tests)
TESTS += $(dlb_tests)
endif

test_common_debug_ldflags = -no-install $(AM_LDFLAGS) $(PTHREAD_CFLAGS) $(PTHREAD_LIBS)
test_common_ldflags = -no-install $(AM_LDFLAGS) $(PTHREAD_CFLAGS) $(PTHREAD_LIBS)

blocking_mercurium_debug_test_SOURCES = ../blocking/blocking.cpp
blocking_mercurium_debug_test_CXXFLAGS = $(DBG_CXXFLAGS) $(AM_CXXFLAGS)
blocking_mercurium_debug_test_LDFLAGS = $(test_common_debug_ldflags)

blocking_mercurium_test_SOURCES = ../blocking/blocking.cpp
blocking_mercurium_test_CPPFLAGS = -DNDEBUG
blocking_mercurium_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
blocking_mercurium_test_LDFLAGS = $(test_common_ldflags)

events_mercurium_debug_test_SOURCES = ../events/events.cpp
events_mercurium_debug_test_CXXFLAGS = $(DBG_CXXFLAGS) $(AM_CXXFLAGS)
events_mercurium_debug_test_LDFLAGS = $(test_common_debug_ldflags)

events_mercurium_test_SOURCES = ../events/events.cpp
events_mercurium_test_CPPFLAGS = -DNDEBUG
events_mercurium_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
events_mercurium_test_LDFLAGS = $(test_common_ldflags)

events_dep_mercurium_debug_test_SOURCES = ../events/events-dep.cpp
events_dep_mercurium_debug_test_CXXFLAGS = $(DBG_CXXFLAGS) $(AM_CXXFLAGS)
events_dep_mercurium_debug_test_LDFLAGS = $(test_common_debug_ldflags)

events_dep_mercurium_test_SOURCES = ../events/events-dep.cpp
events_dep_mercurium_test_CPPFLAGS = -DNDEBUG
events_dep_mercurium_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
events_dep_mercurium_test_LDFLAGS = $(test_common_ldflags)

onready_mercurium_debug_test_SOURCES = ../onready/onready.cpp
onready_mercurium_debug_test_CXXFLAGS = $(DBG_CXXFLAGS) $(AM_CXXFLAGS)
onready_mercurium_debug_test_LDFLAGS = $(test_common_debug_ldflags)

onready_mercurium_test_SOURCES = ../onready/onready.cpp
onready_mercurium_test_CPPFLAGS = -DNDEBUG
onready_mercurium_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
onready_mercurium_test_LDFLAGS = $(test_common_ldflags)

onready_events_mercurium_debug_test_SOURCES = ../onready/onready-events.cpp
onready_events_mercurium_debug_test_CXXFLAGS = $(DBG_CXXFLAGS) $(AM_CXXFLAGS)
onready_events_mercurium_debug_test_LDFLAGS = $(test_common_debug_ldflags)

onready_events_mercurium_test_SOURCES = ../onready/onready-events.cpp
onready_events_mercurium_test_CPPFLAGS = -DNDEBUG
onready_events_mercurium_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
onready_events_mercurium_test_LDFLAGS = $(test_common_ldflags)

scheduling_wait_for_mercurium_debug_test_SOURCES = ../scheduling/scheduling-wait-for.cpp
scheduling_wait_for_mercurium_debug_test_CXXFLAGS = $(DBG_CXXFLAGS) $(AM_CXXFLAGS)
scheduling_wait_for_mercurium_debug_test_LDFLAGS = $(test_common_debug_ldflags)

scheduling_wait_for_mercurium_test_SOURCES = ../scheduling/scheduling-wait-for.cpp
scheduling_wait_for_mercurium_test_CPPFLAGS = -DNDEBUG
scheduling_wait_for_mercurium_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
scheduling_wait_for_mercurium_test_LDFLAGS = $(test_common_ldflags)

fibonacci_mercurium_debug_test_SOURCES = ../fibonacci/fibonacci.cpp
fibonacci_mercurium_debug_test_CXXFLAGS = $(DBG_CXXFLAGS) $(AM_CXXFLAGS)
fibonacci_mercurium_debug_test_LDFLAGS = $(test_common_debug_ldflags)

fibonacci_mercurium_test_SOURCES = ../fibonacci/fibonacci.cpp
fibonacci_mercurium_test_CPPFLAGS = -DNDEBUG
fibonacci_mercurium_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
fibonacci_mercurium_test_LDFLAGS = $(test_common_ldflags)

cpu_activation_mercurium_debug_test_SOURCES = ../cpu-activation/cpu-activation.cpp ../cpu-activation/ConditionVariable.hpp
cpu_activation_mercurium_debug_test_CXXFLAGS = $(DBG_CXXFLAGS) $(AM_CXXFLAGS)
cpu_activation_mercurium_debug_test_LDFLAGS = $(test_common_debug_ldflags)

cpu_activation_mercurium_test_SOURCES = ../cpu-activation/cpu-activation.cpp ../cpu-activation/ConditionVariable.hpp
cpu_activation_mercurium_test_CPPFLAGS = -DNDEBUG
cpu_activation_mercurium_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
cpu_activation_mercurium_test_LDFLAGS = $(test_common_ldflags)

critical_mercurium_debug_test_SOURCES = ../critical/critical.cpp
critical_mercurium_debug_test_CXXFLAGS = $(DBG_CXXFLAGS) $(AM_CXXFLAGS)
critical_mercurium_debug_test_LDFLAGS = $(test_common_debug_ldflags)

critical_mercurium_test_SOURCES = ../critical/critical.cpp
critical_mercurium_test_CPPFLAGS = -DNDEBUG
critical_mercurium_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
critical_mercurium_test_LDFLAGS = $(test_common_ldflags)

dep_nonest_mercurium_debug_test_SOURCES = ../dependencies/dep-nonest.cpp
dep_nonest_mercurium_debug_test_CPPFLAGS =
if HAVE_CONCURRENT_SUPPORT
dep_nonest_mercurium_debug_test_CPPFLAGS += -DHAVE_CONCURRENT_SUPPORT
endif
if HAVE_DLB
dep_nonest_mercurium_debug_test_CPPFLAGS += -DHAVE_DLB
endif
dep_nonest_mercurium_debug_test_CXXFLAGS = $(DBG_CXXFLAGS) $(AM_CXXFLAGS)
dep_nonest_mercurium_debug_test_LDFLAGS = $(test_common_debug_ldflags)

dep_nonest_mercurium_test_SOURCES = ../dependencies/dep-nonest.cpp
dep_nonest_mercurium_test_CPPFLAGS = -DNDEBUG
if HAVE_CONCURRENT_SUPPORT
dep_nonest_mercurium_test_CPPFLAGS += -DHAVE_CONCURRENT_SUPPORT
endif
if HAVE_DLB
dep_nonest_mercurium_test_CPPFLAGS += -DHAVE_DLB
endif
dep_nonest_mercurium_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
dep_nonest_mercurium_test_LDFLAGS = $(test_common_ldflags)

dep_early_release_mercurium_debug_test_SOURCES = ../dependencies/dep-early-release.cpp
dep_early_release_mercurium_debug_test_CXXFLAGS = $(DBG_CXXFLAGS) $(AM_CXXFLAGS)
dep_early_release_mercurium_debug_test_LDFLAGS = $(test_common_debug_ldflags)

dep_early_release_mercurium_test_SOURCES = ../dependencies/dep-early-release.cpp
dep_early_release_mercurium_test_CPPFLAGS = -DNDEBUG
dep_early_release_mercurium_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
dep_early_release_mercurium_test_LDFLAGS = $(test_common_ldflags)

dep_er_and_weak_mercurium_debug_test_SOURCES = ../dependencies/dep-er-and-weak.cpp
dep_er_and_weak_mercurium_debug_test_CXXFLAGS = $(DBG_CXXFLAGS) $(AM_CXXFLAGS)
dep_er_and_weak_mercurium_debug_test_LDFLAGS = $(test_common_debug_ldflags)

dep_er_and_weak_mercurium_test_SOURCES = ../dependencies/dep-er-and-weak.cpp
dep_er_and_weak_mercurium_test_CPPFLAGS = -DNDEBUG
dep_er_and_weak_mercurium_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
dep_er_and_weak_mercurium_test_LDFLAGS = $(test_common_ldflags)

if0_mercurium_debug_test_SOURCES = ../dependencies/if0.cpp
if0_mercurium_debug_test_CXXFLAGS = $(DBG_CXXFLAGS) $(AM_CXXFLAGS)
if0_mercurium_debug_test_LDFLAGS = $(test_common_debug_ldflags)

if0_mercurium_test_SOURCES = ../dependencies/if0.cpp
if0_mercurium_test_CPPFLAGS = -DNDEBUG
if0_mercurium_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
if0_mercurium_test_LDFLAGS = $(test_common_ldflags)

dep_wait_mercurium_debug_test_SOURCES = ../dependencies/dep-wait.cpp
dep_wait_mercurium_debug_test_CXXFLAGS = $(DBG_CXXFLAGS) $(AM_CXXFLAGS)
dep_wait_mercurium_debug_test_LDFLAGS = $(test_common_debug_ldflags)

dep_wait_mercurium_test_SOURCES = ../dependencies/dep-wait.cpp
dep_wait_mercurium_test_CPPFLAGS = -DNDEBUG
dep_wait_mercurium_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
dep_wait_mercurium_test_LDFLAGS = $(test_common_ldflags)

simple_commutative_mercurium_debug_test_SOURCES = ../commutative/simple-commutative.cpp
simple_commutative_mercurium_debug_test_CXXFLAGS = $(DBG_CXXFLAGS) $(AM_CXXFLAGS)
simple_commutative_mercurium_debug_test_LDFLAGS = $(test_common_debug_ldflags)

simple_commutative_mercurium_test_SOURCES = ../commutative/simple-commutative.cpp
simple_commutative_mercurium_test_CPPFLAGS = -DNDEBUG
simple_commutative_mercurium_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
simple_commutative_mercurium_test_LDFLAGS = $(test_common_ldflags)

commutative_stencil_mercurium_debug_test_SOURCES = ../commutative/commutative-stencil.cpp
commutative_stencil_mercurium_debug_test_CXXFLAGS = $(DBG_CXXFLAGS) $(AM_CXXFLAGS)
commutative_stencil_mercurium_debug_test_LDFLAGS = $(test_common_debug_ldflags)

commutative_stencil_mercurium_test_SOURCES = ../commutative/commutative-stencil.cpp
commutative_stencil_mercurium_test_CPPFLAGS = -DNDEBUG
commutative_stencil_mercurium_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
commutative_stencil_mercurium_test_LDFLAGS = $(test_common_ldflags)

taskloop_multiaxpy_mercurium_debug_test_SOURCES = ../taskloop/taskloop-multiaxpy.cpp
taskloop_multiaxpy_mercurium_debug_test_CXXFLAGS = $(DBG_CXXFLAGS) $(AM_CXXFLAGS)
taskloop_multiaxpy_mercurium_debug_test_LDFLAGS = $(test_common_debug_ldflags)

taskloop_multiaxpy_mercurium_test_SOURCES = ../taskloop/taskloop-multiaxpy.cpp
taskloop_multiaxpy_mercurium_test_CPPFLAGS = -DNDEBUG
taskloop_multiaxpy_mercurium_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
taskloop_multiaxpy_mercurium_test_LDFLAGS = $(test_common_ldflags)

taskloop_dep_multiaxpy_mercurium_debug_test_SOURCES = ../taskloop/taskloop-dep-multiaxpy.cpp
taskloop_dep_multiaxpy_mercurium_debug_test_CXXFLAGS = $(DBG_CXXFLAGS) $(AM_CXXFLAGS)
taskloop_dep_multiaxpy_mercurium_debug_test_LDFLAGS = $(test_common_debug_ldflags)

taskloop_dep_multiaxpy_mercurium_test_SOURCES = ../taskloop/taskloop-dep-multiaxpy.cpp
taskloop_dep_multiaxpy_mercurium_test_CPPFLAGS = -DNDEBUG
taskloop_dep_multiaxpy_mercurium_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
taskloop_dep_multiaxpy_mercurium_test_LDFLAGS = $(test_common_ldflags)

taskloop_nested_dep_multiaxpy_mercurium_debug_test_SOURCES = ../taskloop/taskloop-nested-dep-multiaxpy.cpp
taskloop_nested_dep_multiaxpy_mercurium_debug_test_CXXFLAGS = $(DBG_CXXFLAGS) $(AM_CXXFLAGS)
taskloop_nested_dep_multiaxpy_mercurium_debug_test_LDFLAGS = $(test_common_debug_ldflags)

taskloop_nested_dep_multiaxpy_mercurium_test_SOURCES = ../taskloop/taskloop-nested-dep-multiaxpy.cpp
taskloop_nested_dep_multiaxpy_mercurium_test_CPPFLAGS = -DNDEBUG
taskloop_nested_dep_multiaxpy_mercurium_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
taskloop_nested_dep_multiaxpy_mercurium_test_LDFLAGS = $(test_common_ldflags)

taskloop_nonpod_mercurium_debug_test_SOURCES = ../taskloop/taskloop-nonpod.cpp
taskloop_nonpod_mercurium_debug_test_CXXFLAGS = $(DBG_CXXFLAGS) $(AM_CXXFLAGS)
taskloop_nonpod_mercurium_debug_test_LDFLAGS = $(test_common_debug_ldflags)

taskloop_nonpod_mercurium_test_SOURCES = ../taskloop/taskloop-nonpod.cpp
taskloop_nonpod_mercurium_test_CPPFLAGS = -DNDEBUG
taskloop_nonpod_mercurium_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
taskloop_nonpod_mercurium_test_LDFLAGS = $(test_common_ldflags)

taskloop_nqueens_mercurium_debug_test_SOURCES = ../taskloop/taskloop-nqueens.cpp
taskloop_nqueens_mercurium_debug_test_CXXFLAGS = $(DBG_CXXFLAGS) $(AM_CXXFLAGS)
taskloop_nqueens_mercurium_debug_test_LDFLAGS = $(test_common_debug_ldflags)

taskloop_nqueens_mercurium_test_SOURCES = ../taskloop/taskloop-nqueens.cpp
taskloop_nqueens_mercurium_test_CPPFLAGS = -DNDEBUG
taskloop_nqueens_mercurium_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
taskloop_nqueens_mercurium_test_LDFLAGS = $(test_common_ldflags)

taskloop_wait_mercurium_debug_test_SOURCES = ../taskloop/taskloop-wait.cpp
taskloop_wait_mercurium_debug_test_CXXFLAGS = $(DBG_CXXFLAGS) $(AM_CXXFLAGS)
taskloop_wait_mercurium_debug_test_LDFLAGS = $(test_common_debug_ldflags)

taskloop_wait_mercurium_test_SOURCES = ../taskloop/taskloop-wait.cpp
taskloop_wait_mercurium_test_CPPFLAGS = -DNDEBUG
taskloop_wait_mercurium_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
taskloop_wait_mercurium_test_LDFLAGS = $(test_common_ldflags)

discrete_taskloop_multiaxpy_mercurium_debug_test_SOURCES = ../taskloop/taskloop-multiaxpy.cpp
discrete_taskloop_multiaxpy_mercurium_debug_test_CXXFLAGS = $(DBG_CXXFLAGS) $(AM_CXXFLAGS)
discrete_taskloop_multiaxpy_mercurium_debug_test_LDFLAGS = $(test_common_debug_ldflags)

discrete_taskloop_multiaxpy_mercurium_test_SOURCES = ../taskloop/taskloop-multiaxpy.cpp
discrete_taskloop_multiaxpy_mercurium_test_CPPFLAGS = -DNDEBUG
discrete_taskloop_multiaxpy_mercurium_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
discrete_taskloop_multiaxpy_mercurium_test_LDFLAGS = $(test_common_ldflags)

discrete_taskloop_dep_multiaxpy_mercurium_debug_test_SOURCES = ../taskloop/taskloop-dep-multiaxpy.cpp
discrete_taskloop_dep_multiaxpy_mercurium_debug_test_CXXFLAGS = $(DBG_CXXFLAGS) $(AM_CXXFLAGS)
discrete_taskloop_dep_multiaxpy_mercurium_debug_test_LDFLAGS = $(test_common_debug_ldflags)

discrete_taskloop_dep_multiaxpy_mercurium_test_SOURCES = ../taskloop/taskloop-dep-multiaxpy.cpp
discrete_taskloop_dep_multiaxpy_mercurium_test_CPPFLAGS = -DNDEBUG
discrete_taskloop_dep_multiaxpy_mercurium_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
discrete_taskloop_dep_multiaxpy_mercurium_test_LDFLAGS = $(test_common_ldflags)

discrete_taskloop_nested_dep_multiaxpy_mercurium_debug_test_SOURCES = ../taskloop/discrete-taskloop-nested-dep-multiaxpy.cpp
discrete_taskloop_nested_dep_multiaxpy_mercurium_debug_test_CXXFLAGS = $(DBG_CXXFLAGS) $(AM_CXXFLAGS)
discrete_taskloop_nested_dep_multiaxpy_mercurium_debug_test_LDFLAGS = $(test_common_debug_ldflags)

discrete_taskloop_nested_dep_multiaxpy_mercurium_test_SOURCES = ../taskloop/discrete-taskloop-nested-dep-multiaxpy.cpp
discrete_taskloop_nested_dep_multiaxpy_mercurium_test_CPPFLAGS = -DNDEBUG
discrete_taskloop_nested_dep_multiaxpy_mercurium_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
discrete_taskloop_nested_dep_multiaxpy_mercurium_test_LDFLAGS = $(test_common_ldflags)

discrete_taskloop_nonpod_mercurium_debug_test_SOURCES = ../taskloop/taskloop-nonpod.cpp
discrete_taskloop_nonpod_mercurium_debug_test_CXXFLAGS = $(DBG_CXXFLAGS) $(AM_CXXFLAGS)
discrete_taskloop_nonpod_mercurium_debug_test_LDFLAGS = $(test_common_debug_ldflags)

discrete_taskloop_nonpod_mercurium_test_SOURCES = ../taskloop/taskloop-nonpod.cpp
discrete_taskloop_nonpod_mercurium_test_CPPFLAGS = -DNDEBUG
discrete_taskloop_nonpod_mercurium_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
discrete_taskloop_nonpod_mercurium_test_LDFLAGS = $(test_common_ldflags)

discrete_taskloop_nqueens_mercurium_debug_test_SOURCES = ../taskloop/taskloop-nqueens.cpp
discrete_taskloop_nqueens_mercurium_debug_test_CXXFLAGS = $(DBG_CXXFLAGS) $(AM_CXXFLAGS)
discrete_taskloop_nqueens_mercurium_debug_test_LDFLAGS = $(test_common_debug_ldflags)

discrete_taskloop_nqueens_mercurium_test_SOURCES = ../taskloop/taskloop-nqueens.cpp
discrete_taskloop_nqueens_mercurium_test_CPPFLAGS = -DNDEBUG
discrete_taskloop_nqueens_mercurium_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
discrete_taskloop_nqueens_mercurium_test_LDFLAGS = $(test_common_ldflags)

lr_nonest_mercurium_debug_test_SOURCES = ../linear-regions/lr-nonest.cpp
lr_nonest_mercurium_debug_test_CXXFLAGS = $(DBG_CXXFLAGS) $(AM_CXXFLAGS)
lr_nonest_mercurium_debug_test_LDFLAGS = $(test_common_debug_ldflags)

lr_nonest_mercurium_test_SOURCES = ../linear-regions/lr-nonest.cpp
lr_nonest_mercurium_test_CPPFLAGS = -DNDEBUG
lr_nonest_mercurium_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
lr_nonest_mercurium_test_LDFLAGS = $(test_common_ldflags)

lr_nonest_upgrades_mercurium_debug_test_SOURCES = ../linear-regions/lr-nonest-upgrades.cpp
lr_nonest_upgrades_mercurium_debug_test_CXXFLAGS = $(DBG_CXXFLAGS) $(AM_CXXFLAGS)
lr_nonest_upgrades_mercurium_debug_test_LDFLAGS = $(test_common_debug_ldflags)

lr_nonest_upgrades_mercurium_test_SOURCES = ../linear-regions/lr-nonest-upgrades.cpp
lr_nonest_upgrades_mercurium_test_CPPFLAGS = -DNDEBUG
lr_nonest_upgrades_mercurium_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
lr_nonest_upgrades_mercurium_test_LDFLAGS = $(test_common_ldflags)

lr_early_release_mercurium_debug_test_SOURCES = ../linear-regions/lr-early-release.cpp
lr_early_release_mercurium_debug_test_CXXFLAGS = $(DBG_CXXFLAGS) $(AM_CXXFLAGS)
lr_early_release_mercurium_debug_test_LDFLAGS = $(test_common_debug_ldflags)

lr_early_release_mercurium_test_SOURCES = ../linear-regions/lr-early-release.cpp
lr_early_release_mercurium_test_CPPFLAGS = -DNDEBUG
lr_early_release_mercurium_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
lr_early_release_mercurium_test_LDFLAGS = $(test_common_ldflags)

lr_er_and_weak_mercurium_debug_test_SOURCES = ../linear-regions/lr-er-and-weak.cpp
lr_er_and_weak_mercurium_debug_test_CXXFLAGS = $(DBG_CXXFLAGS) $(AM_CXXFLAGS)
lr_er_and_weak_mercurium_debug_test_LDFLAGS = $(test_common_debug_ldflags)

lr_er_and_weak_mercurium_test_SOURCES = ../linear-regions/lr-er-and-weak.cpp
lr_er_and_weak_mercurium_test_CPPFLAGS = -DNDEBUG
lr_er_and_weak_mercurium_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
lr_er_and_weak_mercurium_test_LDFLAGS = $(test_common_ldflags)

lr_release_mercurium_debug_test_SOURCES = ../linear-regions/lr-release.cpp
lr_release_mercurium_debug_test_CXXFLAGS = $(DBG_CXXFLAGS) $(AM_CXXFLAGS)
lr_release_mercurium_debug_test_LDFLAGS = $(test_common_debug_ldflags)

lr_release_mercurium_test_SOURCES = ../linear-regions/lr-release.cpp
lr_release_mercurium_test_CPPFLAGS = -DNDEBUG
lr_release_mercurium_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
lr_release_mercurium_test_LDFLAGS = $(test_common_ldflags)

red_firstprivate_mercurium_debug_test_SOURCES = ../reductions/red-firstprivate.cpp
red_firstprivate_mercurium_debug_test_CXXFLAGS = $(DBG_CXXFLAGS) $(AM_CXXFLAGS)
red_firstprivate_mercurium_debug_test_LDFLAGS = $(test_common_debug_ldflags)

red_firstprivate_mercurium_test_SOURCES = ../reductions/red-firstprivate.cpp
red_firstprivate_mercurium_test_CPPFLAGS = -DNDEBUG
red_firstprivate_mercurium_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
red_firstprivate_mercurium_test_LDFLAGS = $(test_common_ldflags)

red_nonest_mercurium_debug_test_SOURCES = ../reductions/red-nonest.cpp
if HAVE_DLB
red_nonest_mercurium_debug_test_CPPFLAGS = -DHAVE_DLB
endif
red_nonest_mercurium_debug_test_CXXFLAGS = $(DBG_CXXFLAGS) $(AM_CXXFLAGS)
red_nonest_mercurium_debug_test_LDFLAGS = $(test_common_debug_ldflags)

red_nonest_mercurium_test_SOURCES = ../reductions/red-nonest.cpp
red_nonest_mercurium_test_CPPFLAGS = -DNDEBUG
if HAVE_DLB
red_nonest_mercurium_test_CPPFLAGS += -DHAVE_DLB
endif
red_nonest_mercurium_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
red_nonest_mercurium_test_LDFLAGS = $(test_common_ldflags)

red_nest_mercurium_debug_test_SOURCES = ../reductions/red-nest.cpp
red_nest_mercurium_debug_test_CXXFLAGS = $(DBG_CXXFLAGS) $(AM_CXXFLAGS)
red_nest_mercurium_debug_test_LDFLAGS = $(test_common_debug_ldflags)

red_nest_mercurium_test_SOURCES = ../reductions/red-nest.cpp
red_nest_mercurium_test_CPPFLAGS = -DNDEBUG
red_nest_mercurium_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
red_nest_mercurium_test_LDFLAGS = $(test_common_ldflags)

red_nest_other_mercurium_debug_test_SOURCES = ../reductions/red-nest-other.cpp
red_nest_other_mercurium_debug_test_CXXFLAGS = $(DBG_CXXFLAGS) $(AM_CXXFLAGS)
red_nest_other_mercurium_debug_test_LDFLAGS = $(test_common_debug_ldflags)

red_nest_other_mercurium_test_SOURCES = ../reductions/red-nest-other.cpp
red_nest_other_mercurium_test_CPPFLAGS = -DNDEBUG
red_nest_other_mercurium_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
red_nest_other_mercurium_test_LDFLAGS = $(test_common_ldflags)

discrete_deps_mercurium_debug_test_SOURCES = ../discrete/discrete-deps.cpp
discrete_deps_mercurium_debug_test_CXXFLAGS = $(DBG_CXXFLAGS) $(AM_CXXFLAGS)
discrete_deps_mercurium_debug_test_LDFLAGS = $(test_common_debug_ldflags)

discrete_deps_mercurium_test_SOURCES = ../discrete/discrete-deps.cpp
discrete_deps_mercurium_test_CPPFLAGS = -DNDEBUG
discrete_deps_mercurium_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
discrete_deps_mercurium_test_LDFLAGS = $(test_common_ldflags)

discrete_deps_nonest_mercurium_debug_test_SOURCES = ../discrete/discrete-deps-nonest.cpp
if HAVE_CONCURRENT_SUPPORT
discrete_deps_nonest_mercurium_debug_test_CPPFLAGS = -DHAVE_CONCURRENT_SUPPORT
endif
discrete_deps_nonest_mercurium_debug_test_CXXFLAGS = $(DBG_CXXFLAGS) $(AM_CXXFLAGS)
discrete_deps_nonest_mercurium_debug_test_LDFLAGS = $(test_common_debug_ldflags)

discrete_deps_nonest_mercurium_test_SOURCES = ../discrete/discrete-deps-nonest.cpp
discrete_deps_nonest_mercurium_test_CPPFLAGS = -DNDEBUG
if HAVE_CONCURRENT_SUPPORT
discrete_deps_nonest_mercurium_test_CPPFLAGS += -DHAVE_CONCURRENT_SUPPORT
endif
discrete_deps_nonest_mercurium_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
discrete_deps_nonest_mercurium_test_LDFLAGS = $(test_common_ldflags)

discrete_deps_early_release_mercurium_debug_test_SOURCES = ../discrete/discrete-deps-early-release.cpp
discrete_deps_early_release_mercurium_debug_test_CXXFLAGS = $(DBG_CXXFLAGS) $(AM_CXXFLAGS)
discrete_deps_early_release_mercurium_debug_test_LDFLAGS = $(test_common_debug_ldflags)

discrete_deps_early_release_mercurium_test_SOURCES = ../discrete/discrete-deps-early-release.cpp
discrete_deps_early_release_mercurium_test_CPPFLAGS = -DNDEBUG
discrete_deps_early_release_mercurium_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
discrete_deps_early_release_mercurium_test_LDFLAGS = $(test_common_ldflags)

discrete_deps_er_and_weak_mercurium_debug_test_SOURCES = ../discrete/discrete-deps-er-and-weak.cpp
discrete_deps_er_and_weak_mercurium_debug_test_CXXFLAGS = $(DBG_CXXFLAGS) $(AM_CXXFLAGS)
discrete_deps_er_and_weak_mercurium_debug_test_LDFLAGS = $(test_common_debug_ldflags)

discrete_deps_er_and_weak_mercurium_test_SOURCES = ../discrete/discrete-deps-er-and-weak.cpp
discrete_deps_er_and_weak_mercurium_test_CPPFLAGS = -DNDEBUG
discrete_deps_er_and_weak_mercurium_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
discrete_deps_er_and_weak_mercurium_test_LDFLAGS = $(test_common_ldflags)

discrete_deps_wait_mercurium_debug_test_SOURCES = ../discrete/discrete-deps-wait.cpp
discrete_deps_wait_mercurium_debug_test_CXXFLAGS = $(DBG_CXXFLAGS) $(AM_CXXFLAGS)
discrete_deps_wait_mercurium_debug_test_LDFLAGS = $(test_common_debug_ldflags)

discrete_deps_wait_mercurium_test_SOURCES = ../discrete/discrete-deps-wait.cpp
discrete_deps_wait_mercurium_test_CPPFLAGS = -DNDEBUG
discrete_deps_wait_mercurium_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
discrete_deps_wait_mercurium_test_LDFLAGS = $(test_common_ldflags)

discrete_release_mercurium_debug_test_SOURCES = ../discrete/discrete-release.cpp
discrete_release_mercurium_debug_test_CXXFLAGS = $(DBG_CXXFLAGS) $(AM_CXXFLAGS)
discrete_release_mercurium_debug_test_LDFLAGS = $(test_common_debug_ldflags)

discrete_release_mercurium_test_SOURCES = ../discrete/discrete-release.cpp
discrete_release_mercurium_test_CPPFLAGS = -DNDEBUG
discrete_release_mercurium_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
discrete_release_mercurium_test_LDFLAGS = $(test_common_ldflags)

discrete_simple_commutative_mercurium_debug_test_SOURCES = ../discrete/discrete-simple-commutative.cpp
discrete_simple_commutative_mercurium_debug_test_CXXFLAGS = $(DBG_CXXFLAGS) $(AM_CXXFLAGS)
discrete_simple_commutative_mercurium_debug_test_LDFLAGS = $(test_common_debug_ldflags)

discrete_simple_commutative_mercurium_test_SOURCES = ../discrete/discrete-simple-commutative.cpp
discrete_simple_commutative_mercurium_test_CPPFLAGS = -DNDEBUG
discrete_simple_commutative_mercurium_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
discrete_simple_commutative_mercurium_test_LDFLAGS = $(test_common_ldflags)

discrete_red_stress_mercurium_debug_test_SOURCES = ../reductions/red-stress.cpp
discrete_red_stress_mercurium_debug_test_CXXFLAGS = $(DBG_CXXFLAGS) $(AM_CXXFLAGS)
discrete_red_stress_mercurium_debug_test_LDFLAGS = $(test_common_debug_ldflags)

discrete_red_stress_mercurium_test_SOURCES = ../reductions/red-stress.cpp
discrete_red_stress_mercurium_test_CPPFLAGS = -DNDEBUG
discrete_red_stress_mercurium_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
discrete_red_stress_mercurium_test_LDFLAGS = $(test_common_ldflags)

red_nqueens_mercurium_debug_test_SOURCES = ../reductions/red-nqueens.cpp
red_nqueens_mercurium_debug_test_CXXFLAGS = $(DBG_CXXFLAGS) $(AM_CXXFLAGS)
red_nqueens_mercurium_debug_test_LDFLAGS = $(test_common_debug_ldflags)

red_nqueens_mercurium_test_SOURCES = ../reductions/red-nqueens.cpp
red_nqueens_mercurium_test_CPPFLAGS = -DNDEBUG
red_nqueens_mercurium_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
red_nqueens_mercurium_test_LDFLAGS = $(test_common_ldflags)

red_stress_mercurium_debug_test_SOURCES = ../reductions/red-stress.cpp
red_stress_mercurium_debug_test_CXXFLAGS = $(DBG_CXXFLAGS) $(AM_CXXFLAGS)
red_stress_mercurium_debug_test_LDFLAGS = $(test_common_debug_ldflags)

red_stress_mercurium_test_SOURCES = ../reductions/red-stress.cpp
red_stress_mercurium_test_CPPFLAGS = -DNDEBUG
red_stress_mercurium_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
red_stress_mercurium_test_LDFLAGS = $(test_common_ldflags)

dlb_cpu_management_mercurium_test_SOURCES = ../dlb/dlb-cpu-management.cpp
dlb_cpu_management_mercurium_test_CPPFLAGS = -DNDEBUG
dlb_cpu_management_mercurium_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
dlb_cpu_management_mercurium_test_LDFLAGS = $(test_common_ldflags)

dlb_cpu_management_mercurium_debug_test_SOURCES = ../dlb/dlb-cpu-management.cpp
dlb_cpu_management_mercurium_debug_test_CXXFLAGS = $(DBG_CXXFLAGS) $(AM_CXXFLAGS)
dlb_cpu_management_mercurium_debug_test_LDFLAGS = $(test_common_debug_ldflags)

dlb_cpu_sharing_mercurium_test_SOURCES = ../dlb/dlb-cpu-sharing.cpp
dlb_cpu_sharing_mercurium_test_CPPFLAGS = -DNDEBUG
dlb_cpu_sharing_mercurium_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
dlb_cpu_sharing_mercurium_test_LDFLAGS = $(test_common_ldflags)

dlb_cpu_sharing_mercurium_debug_test_SOURCES = ../dlb/dlb-cpu-sharing.cpp
dlb_cpu_sharing_mercurium_debug_test_CXXFLAGS = $(DBG_CXXFLAGS) $(AM_CXXFLAGS)
dlb_cpu_sharing_mercurium_debug_test_LDFLAGS = $(test_common_debug_ldflags)

dlb_cpu_sharing_active_process_mercurium_test_SOURCES = ../dlb/dlb-cpu-sharing-active-process.cpp
dlb_cpu_sharing_active_process_mercurium_test_CPPFLAGS = -DNDEBUG
dlb_cpu_sharing_active_process_mercurium_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
dlb_cpu_sharing_active_process_mercurium_test_LDFLAGS = $(test_common_ldflags)

dlb_cpu_sharing_active_process_mercurium_debug_test_SOURCES = ../dlb/dlb-cpu-sharing-active-process.cpp
dlb_cpu_sharing_active_process_mercurium_debug_test_CXXFLAGS = $(DBG_CXXFLAGS) $(AM_CXXFLAGS)
dlb_cpu_sharing_active_process_mercurium_debug_test_LDFLAGS = $(test_common_debug_ldflags)

dlb_cpu_sharing_passive_process_mercurium_test_SOURCES = ../dlb/dlb-cpu-sharing-passive-process.cpp
dlb_cpu_sharing_passive_process_mercurium_test_CPPFLAGS = -DNDEBUG
dlb_cpu_sharing_passive_process_mercurium_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
dlb_cpu_sharing_passive_process_mercurium_test_LDFLAGS = $(test_common_ldflags)

dlb_cpu_sharing_passive_process_mercurium_debug_test_SOURCES = ../dlb/dlb-cpu-sharing-passive-process.cpp
dlb_cpu_sharing_passive_process_mercurium_debug_test_CXXFLAGS = $(DBG_CXXFLAGS) $(AM_CXXFLAGS)
dlb_cpu_sharing_passive_process_mercurium_debug_test_LDFLAGS = $(test_common_debug_ldflags)

numa_allocations_mercurium_test_SOURCES = ../numa/numa-allocations.cpp
numa_allocations_mercurium_test_CPPFLAGS = -DNDEBUG
numa_allocations_mercurium_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
numa_allocations_mercurium_test_LDFLAGS = $(test_common_ldflags)

numa_allocations_mercurium_debug_test_SOURCES = ../numa/numa-allocations.cpp
numa_allocations_mercurium_debug_test_CXXFLAGS = $(DBG_CXXFLAGS) $(AM_CXXFLAGS)
numa_allocations_mercurium_debug_test_LDFLAGS = $(test_common_debug_ldflags)

numa_auto_mercurium_test_SOURCES = ../numa/numa-auto.cpp
numa_auto_mercurium_test_CPPFLAGS = -DNDEBUG
numa_auto_mercurium_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
numa_auto_mercurium_test_LDFLAGS = $(test_common_ldflags)

numa_auto_mercurium_debug_test_SOURCES = ../numa/numa-auto.cpp
numa_auto_mercurium_debug_test_CXXFLAGS = $(DBG_CXXFLAGS) $(AM_CXXFLAGS)
numa_auto_mercurium_debug_test_LDFLAGS = $(test_common_debug_ldflags)

numa_bitmask_manip_mercurium_test_SOURCES = ../numa/numa-bitmask-manip.cpp
numa_bitmask_manip_mercurium_test_CPPFLAGS = -DNDEBUG
numa_bitmask_manip_mercurium_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
numa_bitmask_manip_mercurium_test_LDFLAGS = $(test_common_ldflags)

numa_bitmask_manip_mercurium_debug_test_SOURCES = ../numa/numa-bitmask-manip.cpp
numa_bitmask_manip_mercurium_debug_test_CXXFLAGS = $(DBG_CXXFLAGS) $(AM_CXXFLAGS)
numa_bitmask_manip_mercurium_debug_test_LDFLAGS = $(test_common_debug_ldflags)

numa_irregular_allocations_mercurium_test_SOURCES = ../numa/numa-irregular-allocations.cpp
numa_irregular_allocations_mercurium_test_CPPFLAGS = -DNDEBUG
numa_irregular_allocations_mercurium_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
numa_irregular_allocations_mercurium_test_LDFLAGS = $(test_common_ldflags)

numa_irregular_allocations_mercurium_debug_test_SOURCES = ../numa/numa-irregular-allocations.cpp
numa_irregular_allocations_mercurium_debug_test_CXXFLAGS = $(DBG_CXXFLAGS) $(AM_CXXFLAGS)
numa_irregular_allocations_mercurium_debug_test_LDFLAGS = $(test_common_debug_ldflags)

numa_off_mercurium_test_SOURCES = ../numa/numa-off.cpp
numa_off_mercurium_test_CPPFLAGS = -DNDEBUG
numa_off_mercurium_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
numa_off_mercurium_test_LDFLAGS = $(test_common_ldflags)

numa_off_mercurium_debug_test_SOURCES = ../numa/numa-off.cpp
numa_off_mercurium_debug_test_CXXFLAGS = $(DBG_CXXFLAGS) $(AM_CXXFLAGS)
numa_off_mercurium_debug_test_LDFLAGS = $(test_common_debug_ldflags)

numa_on_mercurium_test_SOURCES = ../numa/numa-on.cpp
numa_on_mercurium_test_CPPFLAGS = -DNDEBUG
numa_on_mercurium_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
numa_on_mercurium_test_LDFLAGS = $(test_common_ldflags)

numa_on_mercurium_debug_test_SOURCES = ../numa/numa-on.cpp
numa_on_mercurium_debug_test_CXXFLAGS = $(DBG_CXXFLAGS) $(AM_CXXFLAGS)
numa_on_mercurium_debug_test_LDFLAGS = $(test_common_debug_ldflags)

numa_wildcards_mercurium_test_SOURCES = ../numa/numa-wildcards.cpp
numa_wildcards_mercurium_test_CPPFLAGS = -DNDEBUG
numa_wildcards_mercurium_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
numa_wildcards_mercurium_test_LDFLAGS = $(test_common_ldflags)

numa_wildcards_mercurium_debug_test_SOURCES = ../numa/numa-wildcards.cpp
numa_wildcards_mercurium_debug_test_CXXFLAGS = $(DBG_CXXFLAGS) $(AM_CXXFLAGS)
numa_wildcards_mercurium_debug_test_LDFLAGS = $(test_common_debug_ldflags)

if AWK_IS_SANE
TEST_LOG_DRIVER = env AM_TAP_AWK='$(AWK)' LD_LIBRARY_PATH='$(top_builddir)/.libs:${LD_LIBRARY_PATH}' $(SHELL) $(top_srcdir)/tests/select-version.sh $(top_builddir) $(SHELL) $(top_srcdir)/tests/tap-driver.sh
else
TEST_LOG_DRIVER = env AM_TAP_AWK='$(AWK)' LD_LIBRARY_PATH='$(top_builddir)/.libs:${LD_LIBRARY_PATH}' $(SHELL) $(top_srcdir)/tests/select-version.sh $(top_builddir) $(PERL) $(top_srcdir)/tests/tap-driver.pl
endif


build-tests-local: $(check_PROGRAMS)

