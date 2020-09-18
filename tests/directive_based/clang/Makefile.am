#	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
#
#	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)


AM_CXXFLAGS = -I$(top_srcdir)/tests $(PTHREAD_CFLAGS)
AM_LDFLAGS =

CXX = $(NANOS6_CLANGXX)

AM_CXXFLAGS += -fdo-not-use-nanos6 -I$(top_srcdir)/api -I$(top_builddir) -include nanos6.h
AM_LDFLAGS += -L$(top_builddir)/.libs -rpath $(abs_top_builddir)/.libs -Wl,-z,lazy ${jemalloc_LIBS}
LDADD = $(top_builddir)/nanos6-main-wrapper.o $(top_builddir)/libnanos6.la -ldl


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
runtime_handled_reductions_tests =
discrete_tests =
dlb_tests =


if HAVE_NANOS6_CLANG
if BUILD_OPTIMIZED_VARIANT
base_tests += \
	blocking.clang.test \
	events.clang.test \
	events-dep.clang.test \
	scheduling-wait-for.clang.test \
	fibonacci.clang.test \
	dep-nonest.clang.test \
	dep-early-release.clang.test \
	dep-er-and-weak.clang.test \
	if0.clang.test \
	dep-wait.clang.test \
	simple-commutative.clang.test \
	commutative-stencil.clang.test \
	task-for-nonpod.clang.test \
	task-for-nqueens.clang.test \
	taskloop-nonpod.clang.test \
	taskloop-nqueens.clang.test \
	taskloop-for-nonpod.clang.test \
	taskloop-for-nqueens.clang.test


# Ignore CPU Activation test if we have DLB
# NOTE: The order of this tests should never change, new DLB-related
#       tests must be added under these
if HAVE_DLB
dlb_tests += \
	dlb-cpu-management.clang.test \
	dlb-cpu-sharing-active-process.clang.test \
	dlb-cpu-sharing-passive-process.clang.test \
	dlb-cpu-sharing.clang.test
else
cpu_management_tests += \
	cpu-activation.clang.test
endif

linear_region_tests += \
	lr-nonest.clang.test \
	lr-nonest-upgrades.clang.test \
	lr-early-release.clang.test  \
	lr-er-and-weak.clang.test \
	lr-release.clang.test

discrete_tests += \
	discrete-deps-nonest.clang.test \
	discrete-deps-nonest.clang.test \
	discrete-deps-early-release.clang.test \
	discrete-deps-er-and-weak.clang.test \
	discrete-deps-wait.clang.test \
	discrete-simple-commutative.clang.test \
	discrete-taskloop-nonpod.clang.test \
	discrete-taskloop-nqueens.clang.test \
	discrete-taskloop-for-nonpod.clang.test \
	discrete-taskloop-for-nqueens.clang.test
endif

if BUILD_DEBUG_VARIANTS
base_tests +=  \
	blocking.clang.debug.test \
	events.clang.debug.test \
	events-dep.clang.debug.test \
	scheduling-wait-for.clang.debug.test \
	fibonacci.clang.debug.test \
	dep-nonest.clang.debug.test \
	dep-early-release.clang.debug.test \
	dep-er-and-weak.clang.debug.test \
	if0.clang.debug.test \
	dep-wait.clang.debug.test \
	simple-commutative.clang.debug.test \
	commutative-stencil.clang.debug.test \
	task-for-nonpod.clang.debug.test \
	task-for-nqueens.clang.debug.test \
	taskloop-nonpod.clang.debug.test \
	taskloop-nqueens.clang.debug.test \
	taskloop-for-nonpod.clang.debug.test \
	taskloop-for-nqueens.clang.debug.test

# Ignore CPU Activation test if we have DLB for now
if HAVE_DLB
dlb_tests += \
	dlb-cpu-management.clang.debug.test \
	dlb-cpu-sharing-active-process.clang.debug.test \
	dlb-cpu-sharing-passive-process.clang.debug.test \
	dlb-cpu-sharing.clang.debug.test
else
cpu_management_tests += \
	cpu-activation.clang.debug.test
endif

linear_region_tests += \
	lr-nonest.clang.debug.test \
	lr-nonest-upgrades.clang.debug.test \
	lr-early-release.clang.debug.test  \
	lr-er-and-weak.clang.debug.test \
	lr-release.clang.debug.test

discrete_tests += \
	discrete-deps-nonest.clang.debug.test \
	discrete-deps-nonest.clang.debug.test \
	discrete-deps-early-release.clang.debug.test \
	discrete-deps-er-and-weak.clang.debug.test \
	discrete-deps-wait.clang.debug.test \
	discrete-simple-commutative.clang.debug.test \
	discrete-taskloop-nonpod.clang.debug.test \
	discrete-taskloop-nqueens.clang.debug.test \
	discrete-taskloop-for-nonpod.clang.debug.test \
	discrete-taskloop-for-nqueens.clang.debug.test
endif

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

if HAVE_REDUCTIONS_SUPPORT
check_PROGRAMS += $(reductions_tests)
TESTS += $(reductions_tests)
endif

if DISCRETE_DEPENDENCIES
check_PROGRAMS += $(discrete_tests)
TESTS += $(discrete_tests)
endif

if HAVE_DLB
check_PROGRAMS += $(dlb_tests)
TESTS += $(dlb_tests)
endif

test_common_debug_ldflags = -no-install $(AM_LDFLAGS) $(PTHREAD_CFLAGS) $(PTHREAD_LIBS)
test_common_ldflags = -no-install $(AM_LDFLAGS) $(PTHREAD_CFLAGS) $(PTHREAD_LIBS)

# Inhibit linker optimization flag since it is not supported by Mercurium with CUDA
test_cuda_flags = --cuda -fno-lto

blocking_clang_debug_test_SOURCES = ../blocking/blocking.cpp
blocking_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
blocking_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

blocking_clang_test_SOURCES = ../blocking/blocking.cpp
blocking_clang_test_CPPFLAGS = -DNDEBUG
blocking_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
blocking_clang_test_LDFLAGS = $(test_common_debug_ldflags)

events_clang_debug_test_SOURCES = ../events/events.cpp
events_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
events_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

events_clang_test_SOURCES = ../events/events.cpp
events_clang_test_CPPFLAGS = -DNDEBUG
events_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
events_clang_test_LDFLAGS = $(test_common_debug_ldflags)

events_dep_clang_debug_test_SOURCES = ../events/events-dep.cpp
events_dep_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
events_dep_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

events_dep_clang_test_SOURCES = ../events/events-dep.cpp
events_dep_clang_test_CPPFLAGS = -DNDEBUG
events_dep_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
events_dep_clang_test_LDFLAGS = $(test_common_debug_ldflags)

scheduling_wait_for_clang_debug_test_SOURCES = ../scheduling/scheduling-wait-for.cpp
scheduling_wait_for_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
scheduling_wait_for_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

scheduling_wait_for_clang_test_SOURCES = ../scheduling/scheduling-wait-for.cpp
scheduling_wait_for_clang_test_CPPFLAGS = -DNDEBUG
scheduling_wait_for_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
scheduling_wait_for_clang_test_LDFLAGS = $(test_common_debug_ldflags)

fibonacci_clang_debug_test_SOURCES = ../fibonacci/fibonacci.cpp
fibonacci_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
fibonacci_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

fibonacci_clang_test_SOURCES = ../fibonacci/fibonacci.cpp
fibonacci_clang_test_CPPFLAGS = -DNDEBUG
fibonacci_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
fibonacci_clang_test_LDFLAGS = $(test_common_debug_ldflags)

cpu_activation_clang_debug_test_SOURCES = ../cpu-activation/cpu-activation.cpp ../cpu-activation/ConditionVariable.hpp
cpu_activation_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
cpu_activation_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

cpu_activation_clang_test_SOURCES = ../cpu-activation/cpu-activation.cpp ../cpu-activation/ConditionVariable.hpp
cpu_activation_clang_test_CPPFLAGS = -DNDEBUG
cpu_activation_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
cpu_activation_clang_test_LDFLAGS = $(test_common_ldflags)

dep_nonest_clang_debug_test_SOURCES = ../dependencies/dep-nonest.cpp
dep_nonest_clang_debug_test_CPPFLAGS =
if HAVE_CONCURRENT_SUPPORT
dep_nonest_clang_debug_test_CPPFLAGS += -DHAVE_CONCURRENT_SUPPORT
endif
if HAVE_DLB
dep_nonest_clang_debug_test_CPPFLAGS += -DHAVE_DLB
endif
dep_nonest_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
dep_nonest_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

dep_nonest_clang_test_SOURCES = ../dependencies/dep-nonest.cpp
dep_nonest_clang_test_CPPFLAGS = -DNDEBUG
if HAVE_CONCURRENT_SUPPORT
dep_nonest_clang_test_CPPFLAGS += -DHAVE_CONCURRENT_SUPPORT
endif
if HAVE_DLB
dep_nonest_clang_test_CPPFLAGS += -DHAVE_DLB
endif
dep_nonest_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
dep_nonest_clang_test_LDFLAGS = $(test_common_ldflags)

dep_early_release_clang_debug_test_SOURCES = ../dependencies/dep-early-release.cpp
dep_early_release_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
dep_early_release_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

dep_early_release_clang_test_SOURCES = ../dependencies/dep-early-release.cpp
dep_early_release_clang_test_CPPFLAGS = -DNDEBUG
dep_early_release_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
dep_early_release_clang_test_LDFLAGS = $(test_common_ldflags)

dep_er_and_weak_clang_debug_test_SOURCES = ../dependencies/dep-er-and-weak.cpp
dep_er_and_weak_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
dep_er_and_weak_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

dep_er_and_weak_clang_test_SOURCES = ../dependencies/dep-er-and-weak.cpp
dep_er_and_weak_clang_test_CPPFLAGS = -DNDEBUG
dep_er_and_weak_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
dep_er_and_weak_clang_test_LDFLAGS = $(test_common_ldflags)

if0_clang_debug_test_SOURCES = ../dependencies/if0.cpp
if0_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
if0_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

if0_clang_test_SOURCES = ../dependencies/if0.cpp
if0_clang_test_CPPFLAGS = -DNDEBUG
if0_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
if0_clang_test_LDFLAGS = $(test_common_ldflags)

dep_wait_clang_debug_test_SOURCES = ../dependencies/dep-wait.cpp
dep_wait_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
dep_wait_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

dep_wait_clang_test_SOURCES = ../dependencies/dep-wait.cpp
dep_wait_clang_test_CPPFLAGS = -DNDEBUG
dep_wait_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
dep_wait_clang_test_LDFLAGS = $(test_common_ldflags)

simple_commutative_clang_debug_test_SOURCES = ../commutative/simple-commutative.cpp
simple_commutative_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
simple_commutative_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

simple_commutative_clang_test_SOURCES = ../commutative/simple-commutative.cpp
simple_commutative_clang_test_CPPFLAGS = -DNDEBUG
simple_commutative_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
simple_commutative_clang_test_LDFLAGS = $(test_common_ldflags)

commutative_stencil_clang_debug_test_SOURCES = ../commutative/commutative-stencil.cpp
commutative_stencil_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
commutative_stencil_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

commutative_stencil_clang_test_SOURCES = ../commutative/commutative-stencil.cpp
commutative_stencil_clang_test_CPPFLAGS = -DNDEBUG
commutative_stencil_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
commutative_stencil_clang_test_LDFLAGS = $(test_common_ldflags)

task_for_nonpod_clang_debug_test_SOURCES = ../task-for/task-for-nonpod.cpp
task_for_nonpod_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
task_for_nonpod_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

task_for_nonpod_clang_test_SOURCES = ../task-for/task-for-nonpod.cpp
task_for_nonpod_clang_test_CPPFLAGS = -DNDEBUG
task_for_nonpod_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
task_for_nonpod_clang_test_LDFLAGS = $(test_common_ldflags)

task_for_nqueens_clang_debug_test_SOURCES = ../task-for/task-for-nqueens.cpp
task_for_nqueens_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
task_for_nqueens_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

task_for_nqueens_clang_test_SOURCES = ../task-for/task-for-nqueens.cpp
task_for_nqueens_clang_test_CPPFLAGS = -DNDEBUG
task_for_nqueens_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
task_for_nqueens_clang_test_LDFLAGS = $(test_common_ldflags)

taskloop_nonpod_clang_debug_test_SOURCES = ../taskloop/taskloop-nonpod.cpp
taskloop_nonpod_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
taskloop_nonpod_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

taskloop_nonpod_clang_test_SOURCES = ../taskloop/taskloop-nonpod.cpp
taskloop_nonpod_clang_test_CPPFLAGS = -DNDEBUG
taskloop_nonpod_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
taskloop_nonpod_clang_test_LDFLAGS = $(test_common_ldflags)

taskloop_nqueens_clang_debug_test_SOURCES = ../taskloop/taskloop-nqueens.cpp
taskloop_nqueens_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
taskloop_nqueens_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

taskloop_nqueens_clang_test_SOURCES = ../taskloop/taskloop-nqueens.cpp
taskloop_nqueens_clang_test_CPPFLAGS = -DNDEBUG
taskloop_nqueens_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
taskloop_nqueens_clang_test_LDFLAGS = $(test_common_ldflags)

discrete_taskloop_nonpod_clang_debug_test_SOURCES = ../discrete-taskloop/taskloop-nonpod.cpp
discrete_taskloop_nonpod_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
discrete_taskloop_nonpod_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

discrete_taskloop_nonpod_clang_test_SOURCES = ../discrete-taskloop/taskloop-nonpod.cpp
discrete_taskloop_nonpod_clang_test_CPPFLAGS = -DNDEBUG
discrete_taskloop_nonpod_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
discrete_taskloop_nonpod_clang_test_LDFLAGS = $(test_common_ldflags)

discrete_taskloop_nqueens_clang_debug_test_SOURCES = ../discrete-taskloop/taskloop-nqueens.cpp
discrete_taskloop_nqueens_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
discrete_taskloop_nqueens_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

discrete_taskloop_nqueens_clang_test_SOURCES = ../discrete-taskloop/taskloop-nqueens.cpp
discrete_taskloop_nqueens_clang_test_CPPFLAGS = -DNDEBUG
discrete_taskloop_nqueens_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
discrete_taskloop_nqueens_clang_test_LDFLAGS = $(test_common_ldflags)

taskloop_for_nonpod_clang_debug_test_SOURCES = ../taskloop-for/taskloop-for-nonpod.cpp
taskloop_for_nonpod_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
taskloop_for_nonpod_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

taskloop_for_nonpod_clang_test_SOURCES = ../taskloop-for/taskloop-for-nonpod.cpp
taskloop_for_nonpod_clang_test_CPPFLAGS = -DNDEBUG
taskloop_for_nonpod_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
taskloop_for_nonpod_clang_test_LDFLAGS = $(test_common_ldflags)

taskloop_for_nqueens_clang_debug_test_SOURCES = ../taskloop-for/taskloop-for-nqueens.cpp
taskloop_for_nqueens_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
taskloop_for_nqueens_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

taskloop_for_nqueens_clang_test_SOURCES = ../taskloop-for/taskloop-for-nqueens.cpp
taskloop_for_nqueens_clang_test_CPPFLAGS = -DNDEBUG
taskloop_for_nqueens_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
taskloop_for_nqueens_clang_test_LDFLAGS = $(test_common_ldflags)

discrete_taskloop_for_nonpod_clang_debug_test_SOURCES = ../discrete-taskloop-for/taskloop-for-nonpod.cpp
discrete_taskloop_for_nonpod_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
discrete_taskloop_for_nonpod_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

discrete_taskloop_for_nonpod_clang_test_SOURCES = ../discrete-taskloop-for/taskloop-for-nonpod.cpp
discrete_taskloop_for_nonpod_clang_test_CPPFLAGS = -DNDEBUG
discrete_taskloop_for_nonpod_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
discrete_taskloop_for_nonpod_clang_test_LDFLAGS = $(test_common_ldflags)

discrete_taskloop_for_nqueens_clang_debug_test_SOURCES = ../discrete-taskloop-for/taskloop-for-nqueens.cpp
discrete_taskloop_for_nqueens_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
discrete_taskloop_for_nqueens_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

discrete_taskloop_for_nqueens_clang_test_SOURCES = ../discrete-taskloop-for/taskloop-for-nqueens.cpp
discrete_taskloop_for_nqueens_clang_test_CPPFLAGS = -DNDEBUG
discrete_taskloop_for_nqueens_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
discrete_taskloop_for_nqueens_clang_test_LDFLAGS = $(test_common_ldflags)

lr_nonest_clang_debug_test_SOURCES = ../linear-regions/lr-nonest.cpp
lr_nonest_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
lr_nonest_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

lr_nonest_clang_test_SOURCES = ../linear-regions/lr-nonest.cpp
lr_nonest_clang_test_CPPFLAGS = -DNDEBUG
lr_nonest_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
lr_nonest_clang_test_LDFLAGS = $(test_common_ldflags)

lr_nonest_upgrades_clang_debug_test_SOURCES = ../linear-regions/lr-nonest-upgrades.cpp
lr_nonest_upgrades_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
lr_nonest_upgrades_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

lr_nonest_upgrades_clang_test_SOURCES = ../linear-regions/lr-nonest-upgrades.cpp
lr_nonest_upgrades_clang_test_CPPFLAGS = -DNDEBUG
lr_nonest_upgrades_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
lr_nonest_upgrades_clang_test_LDFLAGS = $(test_common_ldflags)

lr_early_release_clang_debug_test_SOURCES = ../linear-regions/lr-early-release.cpp
lr_early_release_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
lr_early_release_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

lr_early_release_clang_test_SOURCES = ../linear-regions/lr-early-release.cpp
lr_early_release_clang_test_CPPFLAGS = -DNDEBUG
lr_early_release_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
lr_early_release_clang_test_LDFLAGS = $(test_common_ldflags)

lr_er_and_weak_clang_debug_test_SOURCES = ../linear-regions/lr-er-and-weak.cpp
lr_er_and_weak_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
lr_er_and_weak_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

lr_er_and_weak_clang_test_SOURCES = ../linear-regions/lr-er-and-weak.cpp
lr_er_and_weak_clang_test_CPPFLAGS = -DNDEBUG
lr_er_and_weak_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
lr_er_and_weak_clang_test_LDFLAGS = $(test_common_ldflags)

lr_release_clang_debug_test_SOURCES = ../linear-regions/lr-release.cpp
lr_release_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
lr_release_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

lr_release_clang_test_SOURCES = ../linear-regions/lr-release.cpp
lr_release_clang_test_CPPFLAGS = -DNDEBUG
lr_release_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
lr_release_clang_test_LDFLAGS = $(test_common_ldflags)

discrete_deps_nonest_clang_debug_test_SOURCES = ../discrete/discrete-deps-nonest.cpp
if HAVE_CONCURRENT_SUPPORT
discrete_deps_nonest_clang_debug_test_CPPFLAGS = -DHAVE_CONCURRENT_SUPPORT
endif
discrete_deps_nonest_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
discrete_deps_nonest_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

discrete_deps_nonest_clang_test_SOURCES = ../discrete/discrete-deps-nonest.cpp
discrete_deps_nonest_clang_test_CPPFLAGS = -DNDEBUG
if HAVE_CONCURRENT_SUPPORT
discrete_deps_nonest_clang_test_CPPFLAGS += -DHAVE_CONCURRENT_SUPPORT
endif
discrete_deps_nonest_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
discrete_deps_nonest_clang_test_LDFLAGS = $(test_common_ldflags)

discrete_deps_early_release_clang_debug_test_SOURCES = ../discrete/discrete-deps-early-release.cpp
discrete_deps_early_release_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
discrete_deps_early_release_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

discrete_deps_early_release_clang_test_SOURCES = ../discrete/discrete-deps-early-release.cpp
discrete_deps_early_release_clang_test_CPPFLAGS = -DNDEBUG
discrete_deps_early_release_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
discrete_deps_early_release_clang_test_LDFLAGS = $(test_common_ldflags)

discrete_deps_er_and_weak_clang_debug_test_SOURCES = ../discrete/discrete-deps-er-and-weak.cpp
discrete_deps_er_and_weak_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
discrete_deps_er_and_weak_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

discrete_deps_er_and_weak_clang_test_SOURCES = ../discrete/discrete-deps-er-and-weak.cpp
discrete_deps_er_and_weak_clang_test_CPPFLAGS = -DNDEBUG
discrete_deps_er_and_weak_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
discrete_deps_er_and_weak_clang_test_LDFLAGS = $(test_common_ldflags)

discrete_deps_wait_clang_debug_test_SOURCES = ../discrete/discrete-deps-wait.cpp
discrete_deps_wait_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
discrete_deps_wait_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

discrete_deps_wait_clang_test_SOURCES = ../discrete/discrete-deps-wait.cpp
discrete_deps_wait_clang_test_CPPFLAGS = -DNDEBUG
discrete_deps_wait_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
discrete_deps_wait_clang_test_LDFLAGS = $(test_common_ldflags)

discrete_simple_commutative_clang_debug_test_SOURCES = ../discrete/discrete-simple-commutative.cpp
discrete_simple_commutative_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
discrete_simple_commutative_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

discrete_simple_commutative_clang_test_SOURCES = ../discrete/discrete-simple-commutative.cpp
discrete_simple_commutative_clang_test_CPPFLAGS = -DNDEBUG
discrete_simple_commutative_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
discrete_simple_commutative_clang_test_LDFLAGS = $(test_common_ldflags)

dlb_cpu_management_clang_test_SOURCES = ../dlb/dlb-cpu-management.cpp
dlb_cpu_management_clang_test_CPPFLAGS = -DNDEBUG
dlb_cpu_management_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
dlb_cpu_management_clang_test_LDFLAGS = $(test_common_ldflags)

dlb_cpu_management_clang_debug_test_SOURCES = ../dlb/dlb-cpu-management.cpp
dlb_cpu_management_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
dlb_cpu_management_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

dlb_cpu_sharing_clang_test_SOURCES = ../dlb/dlb-cpu-sharing.cpp
dlb_cpu_sharing_clang_test_CPPFLAGS = -DNDEBUG
dlb_cpu_sharing_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
dlb_cpu_sharing_clang_test_LDFLAGS = $(test_common_ldflags)

dlb_cpu_sharing_clang_debug_test_SOURCES = ../dlb/dlb-cpu-sharing.cpp
dlb_cpu_sharing_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
dlb_cpu_sharing_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

dlb_cpu_sharing_active_process_clang_test_SOURCES = ../dlb/dlb-cpu-sharing-active-process.cpp
dlb_cpu_sharing_active_process_clang_test_CPPFLAGS = -DNDEBUG
dlb_cpu_sharing_active_process_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
dlb_cpu_sharing_active_process_clang_test_LDFLAGS = $(test_common_ldflags)

dlb_cpu_sharing_active_process_clang_debug_test_SOURCES = ../dlb/dlb-cpu-sharing-active-process.cpp
dlb_cpu_sharing_active_process_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
dlb_cpu_sharing_active_process_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

dlb_cpu_sharing_passive_process_clang_test_SOURCES = ../dlb/dlb-cpu-sharing-passive-process.cpp
dlb_cpu_sharing_passive_process_clang_test_CPPFLAGS = -DNDEBUG
dlb_cpu_sharing_passive_process_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
dlb_cpu_sharing_passive_process_clang_test_LDFLAGS = $(test_common_ldflags)

dlb_cpu_sharing_passive_process_clang_debug_test_SOURCES = ../dlb/dlb-cpu-sharing-passive-process.cpp
dlb_cpu_sharing_passive_process_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
dlb_cpu_sharing_passive_process_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

if AWK_IS_SANE
TEST_LOG_DRIVER = env AM_TAP_AWK='$(AWK)' LD_LIBRARY_PATH='$(top_builddir)/.libs:${LD_LIBRARY_PATH}' $(SHELL) $(top_srcdir)/tests/select-version.sh $(SHELL) $(top_srcdir)/tests/tap-driver.sh
else
TEST_LOG_DRIVER = env AM_TAP_AWK='$(AWK)' LD_LIBRARY_PATH='$(top_builddir)/.libs:${LD_LIBRARY_PATH}' $(SHELL) $(top_srcdir)/tests/select-version.sh $(PERL) $(top_srcdir)/tests/tap-driver.pl
endif


build-tests-local: $(check_PROGRAMS)

