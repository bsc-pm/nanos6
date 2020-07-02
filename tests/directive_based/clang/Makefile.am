#	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
#
#	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)


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
	commutative-stencil.clang.test
#	task-for-multiaxpy.clang.test \
#	task-for-dep-multiaxpy.clang.test \
#	task-for-nonpod.clang.test \
#	task-for-nqueens.clang.test \
#	taskloop-multiaxpy.clang.test \
#	taskloop-dep-multiaxpy.clang.test \
#	taskloop-nested-dep-multiaxpy.clang.test \
#	taskloop-nonpod.clang.test \
#	taskloop-nqueens.clang.test \
#	taskloop-for-multiaxpy.clang.test \
#	taskloop-for-dep-multiaxpy.clang.test \
#	taskloop-for-nested-dep-multiaxpy.clang.test \
#	taskloop-for-nonpod.clang.test \
#	taskloop-for-nqueens.clang.test


if USE_CUDA
# base_tests += cuda-saxpy.clang.test
endif
# Ignore CPU Activation test if we have DLB
# NOTE: The order of this tests should never change, new DLB-related
#       tests must be added under these
if HAVE_DLB
dlb_tests += \
	dlb-cpu-management.clang.test \
	dlb-cpu-sharing-active-process.clang.test \
	dlb-cpu-sharing.clang.test
else
cpu_management_tests += \
	cpu-activation.clang.test
endif

# user_mutex_tests += \
#	critical.clang.test

linear_region_tests += \
	lr-nonest.clang.test \
	lr-nonest-upgrades.clang.test \
	lr-early-release.clang.test  \
	lr-er-and-weak.clang.test
#	lr-release.clang.test

# reductions_tests += \
#	red-firstprivate.clang.test \
#	red-nonest.clang.test \
#	red-nest.clang.test \
#	red-nest-other.clang.test \
#	red-nqueens.clang.test \
#	red-stress.clang.test

discrete_tests += \
	discrete-deps-nonest.clang.test \
	discrete-deps-nonest.clang.test \
	discrete-deps-early-release.clang.test \
	discrete-deps-er-and-weak.clang.test \
	discrete-deps-wait.clang.test \
	discrete-simple-commutative.clang.test
#	discrete-deps.clang.test \
#	discrete-red-stress.clang.test \
#	discrete-taskloop-multiaxpy.clang.test \
#	discrete-taskloop-dep-multiaxpy.clang.test \
#	discrete-taskloop-nested-dep-multiaxpy.clang.test \
#	discrete-taskloop-nonpod.clang.test \
#	discrete-taskloop-nqueens.clang.test \
#	discrete-taskloop-for-multiaxpy.clang.test \
#	discrete-taskloop-for-dep-multiaxpy.clang.test \
#	discrete-taskloop-for-nested-dep-multiaxpy.clang.test \
#	discrete-taskloop-for-nonpod.clang.test \
#	discrete-taskloop-for-nqueens.clang.test

# The following tests are designed for testing reductions implementations where
# the combination is handled by the runtime. They are not enabled at the
# moment, as the current implementation is based on task privatization and the
# combination is handled at compile time.
#runtime_handled_reductions_tests += \
#	red-nest-taskwait.clang.test \
#	red-combine-finish-access.clang.test \
#	red-combine-new-access.clang.test \
#	red-combine-taskwait.clang.test \
#	red-combine-taskwait-mark.clang.test
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
	commutative-stencil.clang.debug.test
#	task-for-multiaxpy.clang.debug.test \
#	task-for-dep-multiaxpy.clang.debug.test \
#	task-for-nonpod.clang.debug.test \
#	task-for-nqueens.clang.debug.test \
#	taskloop-multiaxpy.clang.debug.test \
#	taskloop-dep-multiaxpy.clang.debug.test \
#	taskloop-nested-dep-multiaxpy.clang.debug.test \
#	taskloop-nonpod.clang.debug.test \
#	taskloop-nqueens.clang.debug.test \
#	taskloop-for-multiaxpy.clang.debug.test \
#	taskloop-for-dep-multiaxpy.clang.debug.test \
#	taskloop-for-nested-dep-multiaxpy.clang.debug.test \
#	taskloop-for-nonpod.clang.debug.test \
#	taskloop-for-nqueens.clang.debug.test

if USE_CUDA
# base_tests += cuda-saxpy.clang.debug.test
endif
# Ignore CPU Activation test if we have DLB for now
if HAVE_DLB
dlb_tests += \
	dlb-cpu-management.clang.debug.test \
	dlb-cpu-sharing-active-process.clang.debug.test \
	dlb-cpu-sharing.clang.debug.test
else
cpu_management_tests += \
	cpu-activation.clang.debug.test
endif

# user_mutex_tests += \
#	critical.clang.debug.test

linear_region_tests += \
	lr-nonest.clang.debug.test \
	lr-nonest-upgrades.clang.debug.test \
	lr-early-release.clang.debug.test  \
	lr-er-and-weak.clang.debug.test
#	lr-release.clang.debug.test

# reductions_tests += \
#	red-firstprivate.clang.debug.test \
#	red-nonest.clang.debug.test \
#	red-nest.clang.debug.test \
#	red-nest-other.clang.debug.test \
#	red-nqueens.clang.debug.test \
#	red-stress.clang.debug.test

discrete_tests += \
	discrete-deps-nonest.clang.debug.test \
	discrete-deps-nonest.clang.debug.test \
	discrete-deps-early-release.clang.debug.test \
	discrete-deps-er-and-weak.clang.debug.test \
	discrete-deps-wait.clang.debug.test \
	discrete-simple-commutative.clang.debug.test
#	discrete-deps.clang.debug.test \
#	discrete-red-stress.clang.debug.test \
#	discrete-taskloop-multiaxpy.clang.debug.test \
#	discrete-taskloop-dep-multiaxpy.clang.debug.test \
#	discrete-taskloop-nested-dep-multiaxpy.clang.debug.test \
#	discrete-taskloop-nonpod.clang.debug.test \
#	discrete-taskloop-nqueens.clang.debug.test \
#	discrete-taskloop-for-multiaxpy.clang.debug.test \
#	discrete-taskloop-for-dep-multiaxpy.clang.debug.test \
#	discrete-taskloop-for-nested-dep-multiaxpy.clang.debug.test \
#	discrete-taskloop-for-nonpod.clang.debug.test \
#	discrete-taskloop-for-nqueens.clang.debug.test

# The following tests are designed for testing reductions implementations where
# the combination is handled by the runtime. They are not enabled at the
# moment, as the current implementation is based on task privatization and the
# combination is handled at compile time.
#runtime_handled_reductions_tests += \
#	red-nest-taskwait.clang.debug.test \
#	red-combine-finish-access.clang.debug.test \
#	red-combine-new-access.clang.debug.test \
#	red-combine-taskwait.clang.debug.test \
#	red-combine-taskwait-mark.clang.debug.test
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

# Serialize compilation of tests since Mercurium seems to ignore the --parallel flag
# when compiling CUDA programs
if USE_CUDA
.NOTPARALLEL:
endif

.cu.o:
	$(CXX) -c -o $@ $<

# cuda_saxpy_clang_debug_test_SOURCES = ../cuda/cuda-saxpy.cpp ../cuda/cuda-saxpy.hpp ../cuda/cuda-saxpy-kernel.cu
# cuda_saxpy_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS) $(test_cuda_flags)
# cuda_saxpy_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags) $(test_cuda_flags)

# cuda_saxpy_clang_test_SOURCES = ../cuda/cuda-saxpy.cpp ../cuda/cuda-saxpy.hpp ../cuda/cuda-saxpy-kernel.cu
# cuda_saxpy_clang_test_CPPFLAGS = -DNDEBUG
# cuda_saxpy_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS) $(test_cuda_flags)
# cuda_saxpy_clang_test_LDFLAGS = $(test_common_ldflags) $(test_cuda_flags)

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

# critical_clang_debug_test_SOURCES = ../critical/critical.cpp
# critical_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
# critical_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

# critical_clang_test_SOURCES = ../critical/critical.cpp
# critical_clang_test_CPPFLAGS = -DNDEBUG
# critical_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
# critical_clang_test_LDFLAGS = $(test_common_ldflags)

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

# task_for_multiaxpy_clang_debug_test_SOURCES = ../task-for/task-for-multiaxpy.cpp
# task_for_multiaxpy_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
# task_for_multiaxpy_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

# task_for_multiaxpy_clang_test_SOURCES = ../task-for/task-for-multiaxpy.cpp
# task_for_multiaxpy_clang_test_CPPFLAGS = -DNDEBUG
# task_for_multiaxpy_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
# task_for_multiaxpy_clang_test_LDFLAGS = $(test_common_ldflags)

# task_for_dep_multiaxpy_clang_debug_test_SOURCES = ../task-for/task-for-dep-multiaxpy.cpp
# task_for_dep_multiaxpy_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
# task_for_dep_multiaxpy_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

# task_for_dep_multiaxpy_clang_test_SOURCES = ../task-for/task-for-dep-multiaxpy.cpp
# task_for_dep_multiaxpy_clang_test_CPPFLAGS = -DNDEBUG
# task_for_dep_multiaxpy_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
# task_for_dep_multiaxpy_clang_test_LDFLAGS = $(test_common_ldflags)

# task_for_nonpod_clang_debug_test_SOURCES = ../task-for/task-for-nonpod.cpp
# task_for_nonpod_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
# task_for_nonpod_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

# task_for_nonpod_clang_test_SOURCES = ../task-for/task-for-nonpod.cpp
# task_for_nonpod_clang_test_CPPFLAGS = -DNDEBUG
# task_for_nonpod_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
# task_for_nonpod_clang_test_LDFLAGS = $(test_common_ldflags)

# task_for_nqueens_clang_debug_test_SOURCES = ../task-for/task-for-nqueens.cpp
# task_for_nqueens_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
# task_for_nqueens_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

# task_for_nqueens_clang_test_SOURCES = ../task-for/task-for-nqueens.cpp
# task_for_nqueens_clang_test_CPPFLAGS = -DNDEBUG
# task_for_nqueens_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
# task_for_nqueens_clang_test_LDFLAGS = $(test_common_ldflags)

# taskloop_multiaxpy_clang_debug_test_SOURCES = ../taskloop/taskloop-multiaxpy.cpp
# taskloop_multiaxpy_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
# taskloop_multiaxpy_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

# taskloop_multiaxpy_clang_test_SOURCES = ../taskloop/taskloop-multiaxpy.cpp
# taskloop_multiaxpy_clang_test_CPPFLAGS = -DNDEBUG
# taskloop_multiaxpy_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
# taskloop_multiaxpy_clang_test_LDFLAGS = $(test_common_ldflags)

# taskloop_dep_multiaxpy_clang_debug_test_SOURCES = ../taskloop/taskloop-dep-multiaxpy.cpp
# taskloop_dep_multiaxpy_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
# taskloop_dep_multiaxpy_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

# taskloop_dep_multiaxpy_clang_test_SOURCES = ../taskloop/taskloop-dep-multiaxpy.cpp
# taskloop_dep_multiaxpy_clang_test_CPPFLAGS = -DNDEBUG
# taskloop_dep_multiaxpy_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
# taskloop_dep_multiaxpy_clang_test_LDFLAGS = $(test_common_ldflags)

# taskloop_nested_dep_multiaxpy_clang_debug_test_SOURCES = ../taskloop/taskloop-nested-dep-multiaxpy.cpp
# taskloop_nested_dep_multiaxpy_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
# taskloop_nested_dep_multiaxpy_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

# taskloop_nested_dep_multiaxpy_clang_test_SOURCES = ../taskloop/taskloop-nested-dep-multiaxpy.cpp
# taskloop_nested_dep_multiaxpy_clang_test_CPPFLAGS = -DNDEBUG
# taskloop_nested_dep_multiaxpy_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
# taskloop_nested_dep_multiaxpy_clang_test_LDFLAGS = $(test_common_ldflags)

# taskloop_nonpod_clang_debug_test_SOURCES = ../taskloop/taskloop-nonpod.cpp
# taskloop_nonpod_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
# taskloop_nonpod_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

# taskloop_nonpod_clang_test_SOURCES = ../taskloop/taskloop-nonpod.cpp
# taskloop_nonpod_clang_test_CPPFLAGS = -DNDEBUG
# taskloop_nonpod_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
# taskloop_nonpod_clang_test_LDFLAGS = $(test_common_ldflags)

# taskloop_nqueens_clang_debug_test_SOURCES = ../taskloop/taskloop-nqueens.cpp
# taskloop_nqueens_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
# taskloop_nqueens_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

# taskloop_nqueens_clang_test_SOURCES = ../taskloop/taskloop-nqueens.cpp
# taskloop_nqueens_clang_test_CPPFLAGS = -DNDEBUG
# taskloop_nqueens_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
# taskloop_nqueens_clang_test_LDFLAGS = $(test_common_ldflags)

# discrete_taskloop_multiaxpy_clang_debug_test_SOURCES = ../discrete-taskloop/taskloop-multiaxpy.cpp
# discrete_taskloop_multiaxpy_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
# discrete_taskloop_multiaxpy_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

# discrete_taskloop_multiaxpy_clang_test_SOURCES = ../discrete-taskloop/taskloop-multiaxpy.cpp
# discrete_taskloop_multiaxpy_clang_test_CPPFLAGS = -DNDEBUG
# discrete_taskloop_multiaxpy_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
# discrete_taskloop_multiaxpy_clang_test_LDFLAGS = $(test_common_ldflags)

# discrete_taskloop_dep_multiaxpy_clang_debug_test_SOURCES = ../discrete-taskloop/taskloop-dep-multiaxpy.cpp
# discrete_taskloop_dep_multiaxpy_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
# discrete_taskloop_dep_multiaxpy_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

# discrete_taskloop_dep_multiaxpy_clang_test_SOURCES = ../discrete-taskloop/taskloop-dep-multiaxpy.cpp
# discrete_taskloop_dep_multiaxpy_clang_test_CPPFLAGS = -DNDEBUG
# discrete_taskloop_dep_multiaxpy_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
# discrete_taskloop_dep_multiaxpy_clang_test_LDFLAGS = $(test_common_ldflags)

# discrete_taskloop_nested_dep_multiaxpy_clang_debug_test_SOURCES = ../discrete-taskloop/taskloop-nested-dep-multiaxpy.cpp
# discrete_taskloop_nested_dep_multiaxpy_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
# discrete_taskloop_nested_dep_multiaxpy_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

# discrete_taskloop_nested_dep_multiaxpy_clang_test_SOURCES = ../discrete-taskloop/taskloop-nested-dep-multiaxpy.cpp
# discrete_taskloop_nested_dep_multiaxpy_clang_test_CPPFLAGS = -DNDEBUG
# discrete_taskloop_nested_dep_multiaxpy_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
# discrete_taskloop_nested_dep_multiaxpy_clang_test_LDFLAGS = $(test_common_ldflags)

# discrete_taskloop_nonpod_clang_debug_test_SOURCES = ../discrete-taskloop/taskloop-nonpod.cpp
# discrete_taskloop_nonpod_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
# discrete_taskloop_nonpod_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

# discrete_taskloop_nonpod_clang_test_SOURCES = ../discrete-taskloop/taskloop-nonpod.cpp
# discrete_taskloop_nonpod_clang_test_CPPFLAGS = -DNDEBUG
# discrete_taskloop_nonpod_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
# discrete_taskloop_nonpod_clang_test_LDFLAGS = $(test_common_ldflags)

# discrete_taskloop_nqueens_clang_debug_test_SOURCES = ../discrete-taskloop/taskloop-nqueens.cpp
# discrete_taskloop_nqueens_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
# discrete_taskloop_nqueens_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

# discrete_taskloop_nqueens_clang_test_SOURCES = ../discrete-taskloop/taskloop-nqueens.cpp
# discrete_taskloop_nqueens_clang_test_CPPFLAGS = -DNDEBUG
# discrete_taskloop_nqueens_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
# discrete_taskloop_nqueens_clang_test_LDFLAGS = $(test_common_ldflags)

# taskloop_for_multiaxpy_clang_debug_test_SOURCES = ../taskloop-for/taskloop-for-multiaxpy.cpp
# taskloop_for_multiaxpy_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
# taskloop_for_multiaxpy_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

# taskloop_for_multiaxpy_clang_test_SOURCES = ../taskloop-for/taskloop-for-multiaxpy.cpp
# taskloop_for_multiaxpy_clang_test_CPPFLAGS = -DNDEBUG
# taskloop_for_multiaxpy_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
# taskloop_for_multiaxpy_clang_test_LDFLAGS = $(test_common_ldflags)

# taskloop_for_dep_multiaxpy_clang_debug_test_SOURCES = ../taskloop-for/taskloop-for-dep-multiaxpy.cpp
# taskloop_for_dep_multiaxpy_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
# taskloop_for_dep_multiaxpy_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

# taskloop_for_dep_multiaxpy_clang_test_SOURCES = ../taskloop-for/taskloop-for-dep-multiaxpy.cpp
# taskloop_for_dep_multiaxpy_clang_test_CPPFLAGS = -DNDEBUG
# taskloop_for_dep_multiaxpy_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
# taskloop_for_dep_multiaxpy_clang_test_LDFLAGS = $(test_common_ldflags)

# taskloop_for_nested_dep_multiaxpy_clang_debug_test_SOURCES = ../taskloop-for/taskloop-for-nested-dep-multiaxpy.cpp
# taskloop_for_nested_dep_multiaxpy_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
# taskloop_for_nested_dep_multiaxpy_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

# taskloop_for_nested_dep_multiaxpy_clang_test_SOURCES = ../taskloop-for/taskloop-for-nested-dep-multiaxpy.cpp
# taskloop_for_nested_dep_multiaxpy_clang_test_CPPFLAGS = -DNDEBUG
# taskloop_for_nested_dep_multiaxpy_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
# taskloop_for_nested_dep_multiaxpy_clang_test_LDFLAGS = $(test_common_ldflags)

# taskloop_for_nonpod_clang_debug_test_SOURCES = ../taskloop-for/taskloop-for-nonpod.cpp
# taskloop_for_nonpod_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
# taskloop_for_nonpod_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

# taskloop_for_nonpod_clang_test_SOURCES = ../taskloop-for/taskloop-for-nonpod.cpp
# taskloop_for_nonpod_clang_test_CPPFLAGS = -DNDEBUG
# taskloop_for_nonpod_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
# taskloop_for_nonpod_clang_test_LDFLAGS = $(test_common_ldflags)

# taskloop_for_nqueens_clang_debug_test_SOURCES = ../taskloop-for/taskloop-for-nqueens.cpp
# taskloop_for_nqueens_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
# taskloop_for_nqueens_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

# taskloop_for_nqueens_clang_test_SOURCES = ../taskloop-for/taskloop-for-nqueens.cpp
# taskloop_for_nqueens_clang_test_CPPFLAGS = -DNDEBUG
# taskloop_for_nqueens_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
# taskloop_for_nqueens_clang_test_LDFLAGS = $(test_common_ldflags)

# discrete_taskloop_for_multiaxpy_clang_debug_test_SOURCES = ../discrete-taskloop-for/taskloop-for-multiaxpy.cpp
# discrete_taskloop_for_multiaxpy_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
# discrete_taskloop_for_multiaxpy_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

# discrete_taskloop_for_multiaxpy_clang_test_SOURCES = ../discrete-taskloop-for/taskloop-for-multiaxpy.cpp
# discrete_taskloop_for_multiaxpy_clang_test_CPPFLAGS = -DNDEBUG
# discrete_taskloop_for_multiaxpy_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
# discrete_taskloop_for_multiaxpy_clang_test_LDFLAGS = $(test_common_ldflags)

# discrete_taskloop_for_dep_multiaxpy_clang_debug_test_SOURCES = ../discrete-taskloop-for/taskloop-for-dep-multiaxpy.cpp
# discrete_taskloop_for_dep_multiaxpy_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
# discrete_taskloop_for_dep_multiaxpy_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

# discrete_taskloop_for_dep_multiaxpy_clang_test_SOURCES = ../discrete-taskloop-for/taskloop-for-dep-multiaxpy.cpp
# discrete_taskloop_for_dep_multiaxpy_clang_test_CPPFLAGS = -DNDEBUG
# discrete_taskloop_for_dep_multiaxpy_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
# discrete_taskloop_for_dep_multiaxpy_clang_test_LDFLAGS = $(test_common_ldflags)

# discrete_taskloop_for_nested_dep_multiaxpy_clang_debug_test_SOURCES = ../discrete-taskloop-for/taskloop-for-nested-dep-multiaxpy.cpp
# discrete_taskloop_for_nested_dep_multiaxpy_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
# discrete_taskloop_for_nested_dep_multiaxpy_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

# discrete_taskloop_for_nested_dep_multiaxpy_clang_test_SOURCES = ../discrete-taskloop-for/taskloop-for-nested-dep-multiaxpy.cpp
# discrete_taskloop_for_nested_dep_multiaxpy_clang_test_CPPFLAGS = -DNDEBUG
# discrete_taskloop_for_nested_dep_multiaxpy_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
# discrete_taskloop_for_nested_dep_multiaxpy_clang_test_LDFLAGS = $(test_common_ldflags)

# discrete_taskloop_for_nonpod_clang_debug_test_SOURCES = ../discrete-taskloop-for/taskloop-for-nonpod.cpp
# discrete_taskloop_for_nonpod_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
# discrete_taskloop_for_nonpod_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

# discrete_taskloop_for_nonpod_clang_test_SOURCES = ../discrete-taskloop-for/taskloop-for-nonpod.cpp
# discrete_taskloop_for_nonpod_clang_test_CPPFLAGS = -DNDEBUG
# discrete_taskloop_for_nonpod_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
# discrete_taskloop_for_nonpod_clang_test_LDFLAGS = $(test_common_ldflags)

# discrete_taskloop_for_nqueens_clang_debug_test_SOURCES = ../discrete-taskloop-for/taskloop-for-nqueens.cpp
# discrete_taskloop_for_nqueens_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
# discrete_taskloop_for_nqueens_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

# discrete_taskloop_for_nqueens_clang_test_SOURCES = ../discrete-taskloop-for/taskloop-for-nqueens.cpp
# discrete_taskloop_for_nqueens_clang_test_CPPFLAGS = -DNDEBUG
# discrete_taskloop_for_nqueens_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
# discrete_taskloop_for_nqueens_clang_test_LDFLAGS = $(test_common_ldflags)

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

# lr_release_clang_debug_test_SOURCES = ../linear-regions/lr-release.cpp
# lr_release_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
# lr_release_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

# lr_release_clang_test_SOURCES = ../linear-regions/lr-release.cpp
# lr_release_clang_test_CPPFLAGS = -DNDEBUG
# lr_release_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
# lr_release_clang_test_LDFLAGS = $(test_common_ldflags)

# red_firstprivate_clang_debug_test_SOURCES = ../reductions/red-firstprivate.cpp
# red_firstprivate_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
# red_firstprivate_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

# red_firstprivate_clang_test_SOURCES = ../reductions/red-firstprivate.cpp
# red_firstprivate_clang_test_CPPFLAGS = -DNDEBUG
# red_firstprivate_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
# red_firstprivate_clang_test_LDFLAGS = $(test_common_ldflags)

# red_nonest_clang_debug_test_SOURCES = ../reductions/red-nonest.cpp
if HAVE_DLB
# red_nonest_clang_debug_test_CPPFLAGS = -DHAVE_DLB
endif
# red_nonest_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
# red_nonest_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

# red_nonest_clang_test_SOURCES = ../reductions/red-nonest.cpp
# red_nonest_clang_test_CPPFLAGS = -DNDEBUG
if HAVE_DLB
# red_nonest_clang_test_CPPFLAGS += -DHAVE_DLB
endif
# red_nonest_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
# red_nonest_clang_test_LDFLAGS = $(test_common_ldflags)

# red_nest_clang_debug_test_SOURCES = ../reductions/red-nest.cpp
# red_nest_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
# red_nest_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

# red_nest_clang_test_SOURCES = ../reductions/red-nest.cpp
# red_nest_clang_test_CPPFLAGS = -DNDEBUG
# red_nest_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
# red_nest_clang_test_LDFLAGS = $(test_common_ldflags)

# red_nest_other_clang_debug_test_SOURCES = ../reductions/red-nest-other.cpp
# red_nest_other_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
# red_nest_other_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

# red_nest_other_clang_test_SOURCES = ../reductions/red-nest-other.cpp
# red_nest_other_clang_test_CPPFLAGS = -DNDEBUG
# red_nest_other_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
# red_nest_other_clang_test_LDFLAGS = $(test_common_ldflags)

# discrete_deps_clang_debug_test_SOURCES = ../discrete/discrete-deps.cpp
# discrete_deps_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
# discrete_deps_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

# discrete_deps_clang_test_SOURCES = ../discrete/discrete-deps.cpp
# discrete_deps_clang_test_CPPFLAGS = -DNDEBUG
# discrete_deps_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
# discrete_deps_clang_test_LDFLAGS = $(test_common_ldflags)

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

# discrete_red_stress_clang_debug_test_SOURCES = ../discrete/discrete-red-stress.cpp
# discrete_red_stress_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
# discrete_red_stress_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

# discrete_red_stress_clang_test_SOURCES = ../discrete/discrete-red-stress.cpp
# discrete_red_stress_clang_test_CPPFLAGS = -DNDEBUG
# discrete_red_stress_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
# discrete_red_stress_clang_test_LDFLAGS = $(test_common_ldflags)

#red_nest_taskwait_clang_debug_test_SOURCES = ../reductions/red-nest-taskwait.cpp
#red_nest_taskwait_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
#red_nest_taskwait_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

#red_nest_taskwait_clang_test_SOURCES = ../reductions/red-nest-taskwait.cpp
#red_nest_taskwait_clang_test_CPPFLAGS = -DNDEBUG
#red_nest_taskwait_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
#red_nest_taskwait_clang_test_LDFLAGS = $(test_common_ldflags)

#red_combine_finish_access_clang_debug_test_SOURCES = ../reductions/red-combine-finish-access.cpp
#red_combine_finish_access_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
#red_combine_finish_access_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

#red_combine_finish_access_clang_test_SOURCES = ../reductions/red-combine-finish-access.cpp
#red_combine_finish_access_clang_test_CPPFLAGS = -DNDEBUG
#red_combine_finish_access_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
#red_combine_finish_access_clang_test_LDFLAGS = $(test_common_ldflags)

#red_combine_new_access_clang_debug_test_SOURCES = ../reductions/red-combine-new-access.cpp
#red_combine_new_access_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
#red_combine_new_access_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

#red_combine_new_access_clang_test_SOURCES = ../reductions/red-combine-new-access.cpp
#red_combine_new_access_clang_test_CPPFLAGS = -DNDEBUG
#red_combine_new_access_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
#red_combine_new_access_clang_test_LDFLAGS = $(test_common_ldflags)

#red_combine_taskwait_clang_debug_test_SOURCES = ../reductions/red-combine-taskwait.cpp
#red_combine_taskwait_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
#red_combine_taskwait_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

#red_combine_taskwait_clang_test_SOURCES = ../reductions/red-combine-taskwait.cpp
#red_combine_taskwait_clang_test_CPPFLAGS = -DNDEBUG
#red_combine_taskwait_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
#red_combine_taskwait_clang_test_LDFLAGS = $(test_common_ldflags)

#red_combine_taskwait_mark_clang_debug_test_SOURCES = ../reductions/red-combine-taskwait-mark.cpp
#red_combine_taskwait_mark_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
#red_combine_taskwait_mark_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

#red_combine_taskwait_mark_clang_test_SOURCES = ../reductions/red-combine-taskwait-mark.cpp
#red_combine_taskwait_mark_clang_test_CPPFLAGS = -DNDEBUG
#red_combine_taskwait_mark_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
#red_combine_taskwait_mark_clang_test_LDFLAGS = $(test_common_ldflags)

# red_nqueens_clang_debug_test_SOURCES = ../reductions/red-nqueens.cpp
# red_nqueens_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
# red_nqueens_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

# red_nqueens_clang_test_SOURCES = ../reductions/red-nqueens.cpp
# red_nqueens_clang_test_CPPFLAGS = -DNDEBUG
# red_nqueens_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
# red_nqueens_clang_test_LDFLAGS = $(test_common_ldflags)

# red_stress_clang_debug_test_SOURCES = ../reductions/red-stress.cpp
# red_stress_clang_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
# red_stress_clang_debug_test_LDFLAGS = $(test_common_debug_ldflags)

# red_stress_clang_test_SOURCES = ../reductions/red-stress.cpp
# red_stress_clang_test_CPPFLAGS = -DNDEBUG
# red_stress_clang_test_CXXFLAGS = $(OPT_CLANG_CXXFLAGS) $(AM_CXXFLAGS)
# red_stress_clang_test_LDFLAGS = $(test_common_ldflags)

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

if AWK_IS_SANE
TEST_LOG_DRIVER = env AM_TAP_AWK='$(AWK)' LD_LIBRARY_PATH='$(top_builddir)/.libs:${LD_LIBRARY_PATH}' $(SHELL) $(top_srcdir)/tests/select-version.sh $(SHELL) $(top_srcdir)/tests/tap-driver.sh
else
TEST_LOG_DRIVER = env AM_TAP_AWK='$(AWK)' LD_LIBRARY_PATH='$(top_builddir)/.libs:${LD_LIBRARY_PATH}' $(SHELL) $(top_srcdir)/tests/select-version.sh $(PERL) $(top_srcdir)/tests/tap-driver.pl
endif


build-tests-local: $(check_PROGRAMS)

