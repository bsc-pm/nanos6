#	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
#
#	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)


AM_CXXFLAGS = -I$(top_srcdir)/tests $(PTHREAD_CFLAGS)
AM_LDFLAGS =

CXX = $(NANOS6_MCXX) --parallel

if MCC_SUPORTS_EXTERNAL_INSTALL
AM_CXXFLAGS += --no-default-nanos6-inc --no-default-nanos6-lib -I$(top_srcdir)/api -I$(top_builddir) -include nanos6.h
AM_LDFLAGS += -L$(top_builddir)/.libs -rpath $(abs_top_builddir)/.libs -Wl,-z,lazy
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
runtime_handled_reductions_tests =
discrete_tests =
dlb_tests =


if HAVE_NANOS6_MERCURIUM
if BUILD_OPTIMIZED_VARIANT
base_tests += \
	fibonacci.test \
	dep-nonest.test \
	dep-early-release.test \
	dep-er-and-weak.test \
	if0.test \
	dep-wait.test \
	simple-commutative.test \
	commutative-stencil.test \
	task-for-multiaxpy.test \
	task-for-dep-multiaxpy.test \
	task-for-nonpod.test \
	task-for-nqueens.test \
	taskloop-multiaxpy.test \
	taskloop-dep-multiaxpy.test \
	taskloop-nested-dep-multiaxpy.test \
	taskloop-nonpod.test \
	taskloop-nqueens.test


if USE_CUDA
base_tests += cuda-saxpy.test
endif
# Ignore CPU Activation test if we have DLB
# NOTE: The order of this tests should never change, new DLB-related
#       tests must be added under these
if HAVE_DLB
dlb_tests += \
	dlb-cpu-management.test \
	dlb-cpu-sharing-active-process.test \
	dlb-cpu-sharing-passive-process.test \
	dlb-cpu-sharing.test
else
cpu_management_tests += \
	cpu-activation.test
endif

user_mutex_tests += \
	critical.test

linear_region_tests += \
	lr-nonest.test \
	lr-nonest-upgrades.test \
	lr-early-release.test  \
	lr-er-and-weak.test \
	lr-release.test

reductions_tests += \
	red-firstprivate.test \
	red-nonest.test \
	red-nest.test \
	red-nest-other.test \
	red-nqueens.test \
	red-stress.test

discrete_tests += \
	discrete-deps.test \
	discrete-deps-nonest.test \
	discrete-deps-nonest.test \
	discrete-deps-early-release.test \
	discrete-deps-er-and-weak.test \
	discrete-deps-wait.test \
	discrete-simple-commutative.test \
	discrete-red-stress.test

# The following tests are designed for testing reductions implementations where
# the combination is handled by the runtime. They are not enabled at the
# moment, as the current implementation is based on task privatization and the
# combination is handled at compile time.
#runtime_handled_reductions_tests += \
#	red-nest-taskwait.test \
#	red-combine-finish-access.test \
#	red-combine-new-access.test \
#	red-combine-taskwait.test \
#	red-combine-taskwait-mark.test
endif

if BUILD_DEBUG_VARIANTS
base_tests +=  \
	fibonacci.debug.test \
	dep-nonest.debug.test \
	dep-early-release.debug.test \
	dep-er-and-weak.debug.test \
	if0.debug.test \
	dep-wait.debug.test \
	simple-commutative.debug.test \
	commutative-stencil.debug.test \
	task-for-multiaxpy.debug.test \
	task-for-dep-multiaxpy.debug.test \
	task-for-nonpod.debug.test \
	task-for-nqueens.debug.test \
	taskloop-multiaxpy.debug.test \
	taskloop-dep-multiaxpy.debug.test \
	taskloop-nested-dep-multiaxpy.debug.test \
	taskloop-nonpod.debug.test \
	taskloop-nqueens.debug.test

if USE_CUDA
base_tests += cuda-saxpy.debug.test
endif
# Ignore CPU Activation test if we have DLB for now
if HAVE_DLB
dlb_tests += \
	dlb-cpu-management.debug.test \
	dlb-cpu-sharing-active-process.debug.test \
	dlb-cpu-sharing-passive-process.debug.test \
	dlb-cpu-sharing.debug.test
else
cpu_management_tests += \
	cpu-activation.debug.test
endif

user_mutex_tests += \
	critical.debug.test

linear_region_tests += \
	lr-nonest.debug.test \
	lr-nonest-upgrades.debug.test \
	lr-early-release.debug.test  \
	lr-er-and-weak.debug.test \
	lr-release.debug.test

reductions_tests += \
	red-firstprivate.debug.test \
	red-nonest.debug.test \
	red-nest.debug.test \
	red-nest-other.debug.test \
	red-nqueens.debug.test \
	red-stress.debug.test

discrete_tests += \
	discrete-deps.debug.test \
	discrete-deps-nonest.debug.test \
	discrete-deps-nonest.debug.test \
	discrete-deps-early-release.debug.test \
	discrete-deps-er-and-weak.debug.test \
	discrete-deps-wait.debug.test \
	discrete-simple-commutative.debug.test \
	discrete-red-stress.debug.test

# The following tests are designed for testing reductions implementations where
# the combination is handled by the runtime. They are not enabled at the
# moment, as the current implementation is based on task privatization and the
# combination is handled at compile time.
#runtime_handled_reductions_tests += \
#	red-nest-taskwait.debug.test \
#	red-combine-finish-access.debug.test \
#	red-combine-new-access.debug.test \
#	red-combine-taskwait.debug.test \
#	red-combine-taskwait-mark.debug.test
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

cuda_saxpy_debug_test_SOURCES = cuda/cuda-saxpy.cpp cuda/cuda-saxpy.hpp cuda/cuda-saxpy-kernel.cu
cuda_saxpy_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS) $(test_cuda_flags)
cuda_saxpy_debug_test_LDFLAGS = $(test_common_debug_ldflags) $(test_cuda_flags)

cuda_saxpy_test_SOURCES = cuda/cuda-saxpy.cpp cuda/cuda-saxpy.hpp cuda/cuda-saxpy-kernel.cu
cuda_saxpy_test_CPPFLAGS = -DNDEBUG
cuda_saxpy_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS) $(test_cuda_flags)
cuda_saxpy_test_LDFLAGS = $(test_common_ldflags) $(test_cuda_flags)

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
dep_nonest_debug_test_CPPFLAGS =
if HAVE_CONCURRENT_SUPPORT
dep_nonest_debug_test_CPPFLAGS += -DHAVE_CONCURRENT_SUPPORT
endif
if HAVE_DLB
dep_nonest_debug_test_CPPFLAGS += -DHAVE_DLB
endif
dep_nonest_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
dep_nonest_debug_test_LDFLAGS = $(test_common_debug_ldflags)

dep_nonest_test_SOURCES = dependencies/dep-nonest.cpp
dep_nonest_test_CPPFLAGS = -DNDEBUG
if HAVE_CONCURRENT_SUPPORT
dep_nonest_test_CPPFLAGS += -DHAVE_CONCURRENT_SUPPORT
endif
if HAVE_DLB
dep_nonest_test_CPPFLAGS += -DHAVE_DLB
endif
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

dep_wait_debug_test_SOURCES = dependencies/dep-wait.cpp
dep_wait_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
dep_wait_debug_test_LDFLAGS = $(test_common_debug_ldflags)

dep_wait_test_SOURCES = dependencies/dep-wait.cpp
dep_wait_test_CPPFLAGS = -DNDEBUG
dep_wait_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
dep_wait_test_LDFLAGS = $(test_common_ldflags)

simple_commutative_debug_test_SOURCES = commutative/simple-commutative.cpp
simple_commutative_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
simple_commutative_debug_test_LDFLAGS = $(test_common_debug_ldflags)

simple_commutative_test_SOURCES = commutative/simple-commutative.cpp
simple_commutative_test_CPPFLAGS = -DNDEBUG
simple_commutative_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
simple_commutative_test_LDFLAGS = $(test_common_ldflags)

commutative_stencil_debug_test_SOURCES = commutative/commutative-stencil.cpp
commutative_stencil_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
commutative_stencil_debug_test_LDFLAGS = $(test_common_debug_ldflags)

commutative_stencil_test_SOURCES = commutative/commutative-stencil.cpp
commutative_stencil_test_CPPFLAGS = -DNDEBUG
commutative_stencil_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
commutative_stencil_test_LDFLAGS = $(test_common_ldflags)

task_for_multiaxpy_debug_test_SOURCES = task-for/task-for-multiaxpy.cpp
task_for_multiaxpy_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
task_for_multiaxpy_debug_test_LDFLAGS = $(test_common_debug_ldflags)

task_for_multiaxpy_test_SOURCES = task-for/task-for-multiaxpy.cpp
task_for_multiaxpy_test_CPPFLAGS = -DNDEBUG
task_for_multiaxpy_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
task_for_multiaxpy_test_LDFLAGS = $(test_common_ldflags)

task_for_dep_multiaxpy_debug_test_SOURCES = task-for/task-for-dep-multiaxpy.cpp
task_for_dep_multiaxpy_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
task_for_dep_multiaxpy_debug_test_LDFLAGS = $(test_common_debug_ldflags)

task_for_dep_multiaxpy_test_SOURCES = task-for/task-for-dep-multiaxpy.cpp
task_for_dep_multiaxpy_test_CPPFLAGS = -DNDEBUG
task_for_dep_multiaxpy_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
task_for_dep_multiaxpy_test_LDFLAGS = $(test_common_ldflags)

task_for_nonpod_debug_test_SOURCES = task-for/task-for-nonpod.cpp
task_for_nonpod_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
task_for_nonpod_debug_test_LDFLAGS = $(test_common_debug_ldflags)

task_for_nonpod_test_SOURCES = task-for/task-for-nonpod.cpp
task_for_nonpod_test_CPPFLAGS = -DNDEBUG
task_for_nonpod_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
task_for_nonpod_test_LDFLAGS = $(test_common_ldflags)

task_for_nqueens_debug_test_SOURCES = task-for/task-for-nqueens.cpp
task_for_nqueens_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
task_for_nqueens_debug_test_LDFLAGS = $(test_common_debug_ldflags)

task_for_nqueens_test_SOURCES = task-for/task-for-nqueens.cpp
task_for_nqueens_test_CPPFLAGS = -DNDEBUG
task_for_nqueens_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
task_for_nqueens_test_LDFLAGS = $(test_common_ldflags)

taskloop_multiaxpy_debug_test_SOURCES = taskloop/taskloop-multiaxpy.cpp
taskloop_multiaxpy_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
taskloop_multiaxpy_debug_test_LDFLAGS = $(test_common_debug_ldflags)

taskloop_multiaxpy_test_SOURCES = taskloop/taskloop-multiaxpy.cpp
taskloop_multiaxpy_test_CPPFLAGS = -DNDEBUG
taskloop_multiaxpy_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
taskloop_multiaxpy_test_LDFLAGS = $(test_common_ldflags)

taskloop_dep_multiaxpy_debug_test_SOURCES = taskloop/taskloop-dep-multiaxpy.cpp
taskloop_dep_multiaxpy_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
taskloop_dep_multiaxpy_debug_test_LDFLAGS = $(test_common_debug_ldflags)

taskloop_dep_multiaxpy_test_SOURCES = taskloop/taskloop-dep-multiaxpy.cpp
taskloop_dep_multiaxpy_test_CPPFLAGS = -DNDEBUG
taskloop_dep_multiaxpy_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
taskloop_dep_multiaxpy_test_LDFLAGS = $(test_common_ldflags)

taskloop_nested_dep_multiaxpy_debug_test_SOURCES = taskloop/taskloop-nested-dep-multiaxpy.cpp
taskloop_nested_dep_multiaxpy_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
taskloop_nested_dep_multiaxpy_debug_test_LDFLAGS = $(test_common_debug_ldflags)

taskloop_nested_dep_multiaxpy_test_SOURCES = taskloop/taskloop-nested-dep-multiaxpy.cpp
taskloop_nested_dep_multiaxpy_test_CPPFLAGS = -DNDEBUG
taskloop_nested_dep_multiaxpy_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
taskloop_nested_dep_multiaxpy_test_LDFLAGS = $(test_common_ldflags)

taskloop_nonpod_debug_test_SOURCES = taskloop/taskloop-nonpod.cpp
taskloop_nonpod_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
taskloop_nonpod_debug_test_LDFLAGS = $(test_common_debug_ldflags)

taskloop_nonpod_test_SOURCES = taskloop/taskloop-nonpod.cpp
taskloop_nonpod_test_CPPFLAGS = -DNDEBUG
taskloop_nonpod_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
taskloop_nonpod_test_LDFLAGS = $(test_common_ldflags)

taskloop_nqueens_debug_test_SOURCES = taskloop/taskloop-nqueens.cpp
taskloop_nqueens_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
taskloop_nqueens_debug_test_LDFLAGS = $(test_common_debug_ldflags)

taskloop_nqueens_test_SOURCES = taskloop/taskloop-nqueens.cpp
taskloop_nqueens_test_CPPFLAGS = -DNDEBUG
taskloop_nqueens_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
taskloop_nqueens_test_LDFLAGS = $(test_common_ldflags)

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

lr_release_debug_test_SOURCES = linear-regions/lr-release.cpp
lr_release_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
lr_release_debug_test_LDFLAGS = $(test_common_debug_ldflags)

lr_release_test_SOURCES = linear-regions/lr-release.cpp
lr_release_test_CPPFLAGS = -DNDEBUG
lr_release_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
lr_release_test_LDFLAGS = $(test_common_ldflags)

red_firstprivate_debug_test_SOURCES = reductions/red-firstprivate.cpp
red_firstprivate_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
red_firstprivate_debug_test_LDFLAGS = $(test_common_debug_ldflags)

red_firstprivate_test_SOURCES = reductions/red-firstprivate.cpp
red_firstprivate_test_CPPFLAGS = -DNDEBUG
red_firstprivate_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
red_firstprivate_test_LDFLAGS = $(test_common_ldflags)

red_nonest_debug_test_SOURCES = reductions/red-nonest.cpp
if HAVE_DLB
red_nonest_debug_test_CPPFLAGS = -DHAVE_DLB
endif
red_nonest_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
red_nonest_debug_test_LDFLAGS = $(test_common_debug_ldflags)

red_nonest_test_SOURCES = reductions/red-nonest.cpp
red_nonest_test_CPPFLAGS = -DNDEBUG
if HAVE_DLB
red_nonest_test_CPPFLAGS += -DHAVE_DLB
endif
red_nonest_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
red_nonest_test_LDFLAGS = $(test_common_ldflags)

red_nest_debug_test_SOURCES = reductions/red-nest.cpp
red_nest_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
red_nest_debug_test_LDFLAGS = $(test_common_debug_ldflags)

red_nest_test_SOURCES = reductions/red-nest.cpp
red_nest_test_CPPFLAGS = -DNDEBUG
red_nest_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
red_nest_test_LDFLAGS = $(test_common_ldflags)

red_nest_other_debug_test_SOURCES = reductions/red-nest-other.cpp
red_nest_other_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
red_nest_other_debug_test_LDFLAGS = $(test_common_debug_ldflags)

red_nest_other_test_SOURCES = reductions/red-nest-other.cpp
red_nest_other_test_CPPFLAGS = -DNDEBUG
red_nest_other_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
red_nest_other_test_LDFLAGS = $(test_common_ldflags)

discrete_deps_debug_test_SOURCES = discrete/discrete-deps.cpp
discrete_deps_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
discrete_deps_debug_test_LDFLAGS = $(test_common_debug_ldflags)

discrete_deps_test_SOURCES = discrete/discrete-deps.cpp
discrete_deps_test_CPPFLAGS = -DNDEBUG
discrete_deps_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
discrete_deps_test_LDFLAGS = $(test_common_ldflags)

discrete_deps_nonest_debug_test_SOURCES = discrete/discrete-deps-nonest.cpp
if HAVE_CONCURRENT_SUPPORT
discrete_deps_nonest_debug_test_CPPFLAGS = -DHAVE_CONCURRENT_SUPPORT
endif
discrete_deps_nonest_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
discrete_deps_nonest_debug_test_LDFLAGS = $(test_common_debug_ldflags)

discrete_deps_nonest_test_SOURCES = discrete/discrete-deps-nonest.cpp
discrete_deps_nonest_test_CPPFLAGS = -DNDEBUG
if HAVE_CONCURRENT_SUPPORT
discrete_deps_nonest_test_CPPFLAGS += -DHAVE_CONCURRENT_SUPPORT
endif
discrete_deps_nonest_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
discrete_deps_nonest_test_LDFLAGS = $(test_common_ldflags)

discrete_deps_early_release_debug_test_SOURCES = discrete/discrete-deps-early-release.cpp
discrete_deps_early_release_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
discrete_deps_early_release_debug_test_LDFLAGS = $(test_common_debug_ldflags)

discrete_deps_early_release_test_SOURCES = discrete/discrete-deps-early-release.cpp
discrete_deps_early_release_test_CPPFLAGS = -DNDEBUG
discrete_deps_early_release_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
discrete_deps_early_release_test_LDFLAGS = $(test_common_ldflags)

discrete_deps_er_and_weak_debug_test_SOURCES = discrete/discrete-deps-er-and-weak.cpp
discrete_deps_er_and_weak_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
discrete_deps_er_and_weak_debug_test_LDFLAGS = $(test_common_debug_ldflags)

discrete_deps_er_and_weak_test_SOURCES = discrete/discrete-deps-er-and-weak.cpp
discrete_deps_er_and_weak_test_CPPFLAGS = -DNDEBUG
discrete_deps_er_and_weak_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
discrete_deps_er_and_weak_test_LDFLAGS = $(test_common_ldflags)

discrete_deps_wait_debug_test_SOURCES = discrete/discrete-deps-wait.cpp
discrete_deps_wait_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
discrete_deps_wait_debug_test_LDFLAGS = $(test_common_debug_ldflags)

discrete_deps_wait_test_SOURCES = discrete/discrete-deps-wait.cpp
discrete_deps_wait_test_CPPFLAGS = -DNDEBUG
discrete_deps_wait_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
discrete_deps_wait_test_LDFLAGS = $(test_common_ldflags)

discrete_simple_commutative_debug_test_SOURCES = discrete/discrete-simple-commutative.cpp
discrete_simple_commutative_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
discrete_simple_commutative_debug_test_LDFLAGS = $(test_common_debug_ldflags)

discrete_simple_commutative_test_SOURCES = discrete/discrete-simple-commutative.cpp
discrete_simple_commutative_test_CPPFLAGS = -DNDEBUG
discrete_simple_commutative_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
discrete_simple_commutative_test_LDFLAGS = $(test_common_ldflags)

discrete_red_stress_debug_test_SOURCES = discrete/discrete-red-stress.cpp
discrete_red_stress_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
discrete_red_stress_debug_test_LDFLAGS = $(test_common_debug_ldflags)

discrete_red_stress_test_SOURCES = discrete/discrete-red-stress.cpp
discrete_red_stress_test_CPPFLAGS = -DNDEBUG
discrete_red_stress_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
discrete_red_stress_test_LDFLAGS = $(test_common_ldflags)

#red_nest_taskwait_debug_test_SOURCES = reductions/red-nest-taskwait.cpp
#red_nest_taskwait_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
#red_nest_taskwait_debug_test_LDFLAGS = $(test_common_debug_ldflags)

#red_nest_taskwait_test_SOURCES = reductions/red-nest-taskwait.cpp
#red_nest_taskwait_test_CPPFLAGS = -DNDEBUG
#red_nest_taskwait_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
#red_nest_taskwait_test_LDFLAGS = $(test_common_ldflags)

#red_combine_finish_access_debug_test_SOURCES = reductions/red-combine-finish-access.cpp
#red_combine_finish_access_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
#red_combine_finish_access_debug_test_LDFLAGS = $(test_common_debug_ldflags)

#red_combine_finish_access_test_SOURCES = reductions/red-combine-finish-access.cpp
#red_combine_finish_access_test_CPPFLAGS = -DNDEBUG
#red_combine_finish_access_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
#red_combine_finish_access_test_LDFLAGS = $(test_common_ldflags)

#red_combine_new_access_debug_test_SOURCES = reductions/red-combine-new-access.cpp
#red_combine_new_access_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
#red_combine_new_access_debug_test_LDFLAGS = $(test_common_debug_ldflags)

#red_combine_new_access_test_SOURCES = reductions/red-combine-new-access.cpp
#red_combine_new_access_test_CPPFLAGS = -DNDEBUG
#red_combine_new_access_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
#red_combine_new_access_test_LDFLAGS = $(test_common_ldflags)

#red_combine_taskwait_debug_test_SOURCES = reductions/red-combine-taskwait.cpp
#red_combine_taskwait_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
#red_combine_taskwait_debug_test_LDFLAGS = $(test_common_debug_ldflags)

#red_combine_taskwait_test_SOURCES = reductions/red-combine-taskwait.cpp
#red_combine_taskwait_test_CPPFLAGS = -DNDEBUG
#red_combine_taskwait_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
#red_combine_taskwait_test_LDFLAGS = $(test_common_ldflags)

#red_combine_taskwait_mark_debug_test_SOURCES = reductions/red-combine-taskwait-mark.cpp
#red_combine_taskwait_mark_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
#red_combine_taskwait_mark_debug_test_LDFLAGS = $(test_common_debug_ldflags)

#red_combine_taskwait_mark_test_SOURCES = reductions/red-combine-taskwait-mark.cpp
#red_combine_taskwait_mark_test_CPPFLAGS = -DNDEBUG
#red_combine_taskwait_mark_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
#red_combine_taskwait_mark_test_LDFLAGS = $(test_common_ldflags)

red_nqueens_debug_test_SOURCES = reductions/red-nqueens.cpp
red_nqueens_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
red_nqueens_debug_test_LDFLAGS = $(test_common_debug_ldflags)

red_nqueens_test_SOURCES = reductions/red-nqueens.cpp
red_nqueens_test_CPPFLAGS = -DNDEBUG
red_nqueens_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
red_nqueens_test_LDFLAGS = $(test_common_ldflags)

red_stress_debug_test_SOURCES = reductions/red-stress.cpp
red_stress_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
red_stress_debug_test_LDFLAGS = $(test_common_debug_ldflags)

red_stress_test_SOURCES = reductions/red-stress.cpp
red_stress_test_CPPFLAGS = -DNDEBUG
red_stress_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
red_stress_test_LDFLAGS = $(test_common_ldflags)

dlb_cpu_management_test_SOURCES = dlb/dlb-cpu-management.cpp
dlb_cpu_management_test_CPPFLAGS = -DNDEBUG
dlb_cpu_management_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
dlb_cpu_management_test_LDFLAGS = $(test_common_ldflags)

dlb_cpu_management_debug_test_SOURCES = dlb/dlb-cpu-management.cpp
dlb_cpu_management_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
dlb_cpu_management_debug_test_LDFLAGS = $(test_common_debug_ldflags)

dlb_cpu_sharing_test_SOURCES = dlb/dlb-cpu-sharing.cpp
dlb_cpu_sharing_test_CPPFLAGS = -DNDEBUG
dlb_cpu_sharing_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
dlb_cpu_sharing_test_LDFLAGS = $(test_common_ldflags)

dlb_cpu_sharing_debug_test_SOURCES = dlb/dlb-cpu-sharing.cpp
dlb_cpu_sharing_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
dlb_cpu_sharing_debug_test_LDFLAGS = $(test_common_debug_ldflags)

dlb_cpu_sharing_active_process_test_SOURCES = dlb/dlb-cpu-sharing-active-process.cpp
dlb_cpu_sharing_active_process_test_CPPFLAGS = -DNDEBUG
dlb_cpu_sharing_active_process_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
dlb_cpu_sharing_active_process_test_LDFLAGS = $(test_common_ldflags)

dlb_cpu_sharing_active_process_debug_test_SOURCES = dlb/dlb-cpu-sharing-active-process.cpp
dlb_cpu_sharing_active_process_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
dlb_cpu_sharing_active_process_debug_test_LDFLAGS = $(test_common_debug_ldflags)

dlb_cpu_sharing_passive_process_test_SOURCES = dlb/dlb-cpu-sharing-passive-process.cpp
dlb_cpu_sharing_passive_process_test_CPPFLAGS = -DNDEBUG
dlb_cpu_sharing_passive_process_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
dlb_cpu_sharing_passive_process_test_LDFLAGS = $(test_common_ldflags)

dlb_cpu_sharing_passive_process_debug_test_SOURCES = dlb/dlb-cpu-sharing-passive-process.cpp
dlb_cpu_sharing_passive_process_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
dlb_cpu_sharing_passive_process_debug_test_LDFLAGS = $(test_common_debug_ldflags)


if AWK_IS_SANE
TEST_LOG_DRIVER = env AM_TAP_AWK='$(AWK)' LD_LIBRARY_PATH='$(top_builddir)/.libs:${LD_LIBRARY_PATH}' $(SHELL) $(top_srcdir)/tests/select-version.sh $(SHELL) $(top_srcdir)/tests/tap-driver.sh
else
TEST_LOG_DRIVER = env AM_TAP_AWK='$(AWK)' LD_LIBRARY_PATH='$(top_builddir)/.libs:${LD_LIBRARY_PATH}' $(SHELL) $(top_srcdir)/tests/select-version.sh $(PERL) $(top_srcdir)/tests/tap-driver.pl
endif


build-tests-local: $(check_PROGRAMS)

