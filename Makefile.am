#	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
#	
#	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)

ACLOCAL_AMFLAGS = -I m4
AM_CFLAGS = -I$(srcdir)/api -I$(srcdir)/loader -I.
AM_CXXFLAGS = -I$(srcdir)/src -I$(builddir)/src -I$(srcdir)/api -I.
AM_LDFLAGS = $(AS_NEEDED_FLAGS) 

SUBDIRS = . commands tests/directive_based scripts


# See info page of libtool "Updating version info"
# https://www.gnu.org/software/libtool/manual/html_node/Updating-version-info.html
lib_current = 0
lib_revision = 0
lib_age = 0

# Expected shared object numbers
lib_major = $(shell expr $(lib_current) - $(lib_age))
lib_suffix = $(lib_major).$(lib_age).$(lib_revision)


MAX_REGION_DIMENSIONS = 8
MULTIDIMENSIONAL_REGION_API_TYPES = read write readwrite weak_read weak_write weak_readwrite concurrent commutative reduction
SUPPORTED_MULTIDIMENSIONAL_REGION_API_TYPES = 

nanos6includedir = $(includedir)/nanos6

nanos6include_HEADERS = \
	api/nanos6/blocking.h \
	api/nanos6/bootstrap.h \
	api/nanos6/constants.h \
	api/nanos6/debug.h \
	api/nanos6/dependencies.h \
	api/nanos6/final.h \
	api/nanos6/library-mode.h \
	api/nanos6/polling.h \
	api/nanos6/reductions.h \
	api/nanos6/runtime-info.h \
	api/nanos6/task-info-registration.h \
	api/nanos6/task-instantiation.h \
	api/nanos6/taskwait.h \
	api/nanos6/user-mutex.h \
	api/nanos6/utils.h

nodist_nanos6include_HEADERS = \
	nanos6/multidimensional-dependencies.h \
	nanos6/multidimensional-release.h 

include_HEADERS = \
	api/nanos6.h


lib_LTLIBRARIES = \
	libnanos6.la \
	libnanos6-optimized.la libnanos6-debug.la \
	libnanos6-graph.la libnanos6-graph-debug.la \
	libnanos6-profile.la \
	libnanos6-stats.la \
	libnanos6-verbose.la libnanos6-verbose-debug.la

lib_LIBRARIES = libnanos6-main-wrapper.a


#
# Taskification of the "main" function
#
main_interception =
if LINUX_POWERPC_GLIBC
main_interception += loader/intercept-main-glibc-powerpc.c
endif
if LINUX_GLIBC
main_interception += loader/intercept-main-glibc.c
endif
if ANDROID
main_interception += loader/intercept-main-android.c
endif

libnanos6_main_wrapper_a_SOURCES = \
	$(main_interception) \
	loader/error.h \
	loader/function-interception.h \
	loader/intercept-main-common.c \
	loader/intercept-main-common.h \
	loader/main-wrapper.c \
	loader/main-wrapper.h \
	loader/malloc.h


#
# Loader and symbol resolution
#
symbol_resolution =
symbol_resolution_header =
multidimensional_regions = 

if RESOLVE_SYMBOLS_USING_IFUNC
symbol_resolution_header += loader/symbol-resolver/resolve.h
symbol_resolution += \
	loader/indirect-symbols/malloc.c \
	loader/symbol-resolver/blocking.c \
	loader/symbol-resolver/bootstrap.c \
	loader/symbol-resolver/cpu-control.c \
	loader/symbol-resolver/debugging.c \
	loader/symbol-resolver/dependencies.c \
	loader/symbol-resolver/final.c \
	loader/symbol-resolver/polling.c \
	loader/symbol-resolver/reductions.c \
	loader/symbol-resolver/runtime-info.c \
	loader/symbol-resolver/task-info-registration.c \
	loader/symbol-resolver/task-instantiation.c \
	loader/symbol-resolver/taskwait.c \
	loader/symbol-resolver/user-mutex.c \
	loader/symbol-resolver/utils.c \
	loader/symbol-resolver/weak-dependencies.c
multidimensional_regions += \
	multidim-region-dependency-resolvers.c \
	multidim-release-resolvers.c
endif


if RESOLVE_SYMBOLS_USING_INDIRECTION
symbol_resolution_header += loader/indirect-symbols/resolve.h
symbol_resolution += \
	loader/indirect-symbols/blocking.c \
	loader/indirect-symbols/bootstrap.c \
	loader/indirect-symbols/cpu-control.c \
	loader/indirect-symbols/debugging.c \
	loader/indirect-symbols/dependencies.c \
	loader/indirect-symbols/final.c \
	loader/indirect-symbols/malloc.c \
	loader/indirect-symbols/polling.c \
	loader/indirect-symbols/reductions.c \
	loader/indirect-symbols/runtime-info.c \
	loader/indirect-symbols/task-info-registration.c \
	loader/indirect-symbols/task-instantiation.c \
	loader/indirect-symbols/taskwait.c \
	loader/indirect-symbols/user-mutex.c \
	loader/indirect-symbols/utils.c \
	loader/indirect-symbols/weak-dependencies.c

multidimensional_regions += \
	multidim-region-dependency-indirect-resolvers.c \
	multidim-release-indirect-resolvers.c

dependent_objects = $(symbol_resolution:.c=.lo) $(multidimensional_regions:.c=.lo)
$(dependent_objects): nanos6/multidimensional-dependencies.h
endif

libnanos6_la_SOURCES = \
	$(symbol_resolution_header) \
	$(symbol_resolution) \
	loader/error.h \
	loader/loader.c \
	loader/loader.h

libnanos6_la_CPPFLAGS = -DSONAME_MAJOR=\"$(lib_major)\" -DSONAME_SUFFIX=\"$(lib_suffix)\"
libnanos6_la_LDFLAGS = $(AM_LDFLAGS) $(PTHREAD_CFLAGS) $(PTHREAD_LIBS) $(DLOPEN_LIBS) $(LDFLAGS_NOUNDEFINED)


nanos6_generated_headers = \
	nanos6/multidimensional-dependencies.h \
	nanos6/multidimensional-release.h
	
nodist_libnanos6_la_SOURCES = \
	$(nanos6_generated_headers) \
	multidim-region-dependency-fallbacks.h \
	multidim-region-dependency-fallbacks.c \
	multidim-release-fallbacks.h \
	multidim-release-fallbacks.c \
	$(multidimensional_regions)
CLEANFILES = $(nodist_libnanos6_la_SOURCES)

BUILT_SOURCES = \
	$(nanos6_generated_headers)


#
# Automatically generated API headers
#
EXTRA_DIST = \
	loader/scripts/common.sh \
	loader/scripts/generate_multidim_functions_for_linear_regions.sh \
	loader/scripts/generate_regions_fallback_functions.sh \
	loader/scripts/generate_regions_fallback_prototypes.sh \
	loader/scripts/generate_regions_indirect_symbol_resolvers.sh \
	loader/scripts/generate_regions_prototypes.sh \
	loader/scripts/generate_regions_symbol_resolvers.sh \
	loader/scripts/generate_release_fallback_functions.sh \
	loader/scripts/generate_release_fallback_prototypes.sh \
	loader/scripts/generate_release_indirect_symbol_resolvers.sh \
	loader/scripts/generate_release_prototypes.sh \
	loader/scripts/generate_release_symbol_resolvers.sh


AM_V_GEN = $(am__v_GEN_@AM_V@)
am__v_GEN_ = $(am__v_GEN_@AM_DEFAULT_V@)
am__v_GEN_0 = @echo "  GEN     " $@;
am__v_GEN_1 =

AM_V_GEN2 = $(am__v_GEN2_@AM_V@)
am__v_GEN2_ = $(am__v_GEN2_@AM_DEFAULT_V@)
am__v_GEN2_0 = @echo "  GEN2    " $@;
am__v_GEN2_1 =

AM_V_GEN3 = $(am__v_GEN3_@AM_V@)
am__v_GEN3_ = $(am__v_GEN3_@AM_DEFAULT_V@)
am__v_GEN3_0 = @echo "  GEN3    " $@;
am__v_GEN3_1 =

nanos6/multidimensional-dependencies.h: $(top_srcdir)/loader/scripts/common.sh $(top_srcdir)/loader/scripts/generate_regions_prototypes.sh Makefile
	$(AM_V_GEN)mkdir -p nanos6 ; \
	$(top_srcdir)/loader/scripts/generate_regions_prototypes.sh $(MAX_REGION_DIMENSIONS) $(MULTIDIMENSIONAL_REGION_API_TYPES) > $@

multidim-region-dependency-fallbacks.h: $(top_srcdir)/loader/scripts/common.sh $(top_srcdir)/loader/scripts/generate_regions_fallback_prototypes.sh Makefile
	$(AM_V_GEN)$(top_srcdir)/loader/scripts/generate_regions_fallback_prototypes.sh $(MAX_REGION_DIMENSIONS) $(MULTIDIMENSIONAL_REGION_API_TYPES) > $@

multidim-region-dependency-fallbacks.c: $(top_srcdir)/loader/scripts/common.sh $(top_srcdir)/loader/scripts/generate_regions_fallback_functions.sh Makefile multidim-region-dependency-fallbacks.h nanos6/multidimensional-dependencies.h
	$(AM_V_GEN)$(top_srcdir)/loader/scripts/generate_regions_fallback_functions.sh $(MAX_REGION_DIMENSIONS) $(MULTIDIMENSIONAL_REGION_API_TYPES) > $@

multidim-region-dependency-resolvers.c: $(top_srcdir)/loader/scripts/common.sh $(top_srcdir)/loader/scripts/generate_regions_symbol_resolvers.sh Makefile multidim-region-dependency-fallbacks.h
	$(AM_V_GEN)$(top_srcdir)/loader/scripts/generate_regions_symbol_resolvers.sh $(MAX_REGION_DIMENSIONS) $(MULTIDIMENSIONAL_REGION_API_TYPES) > $@

multidim-region-dependency-indirect-resolvers.c: $(top_srcdir)/loader/scripts/common.sh $(top_srcdir)/loader/scripts/generate_regions_indirect_symbol_resolvers.sh Makefile multidim-region-dependency-fallbacks.h nanos6/multidimensional-dependencies.h
	$(AM_V_GEN)$(top_srcdir)/loader/scripts/generate_regions_indirect_symbol_resolvers.sh $(MAX_REGION_DIMENSIONS) $(MULTIDIMENSIONAL_REGION_API_TYPES) > $@

nanos6/multidimensional-release.h: $(top_srcdir)/loader/scripts/common.sh $(top_srcdir)/loader/scripts/generate_release_prototypes.sh Makefile
	$(AM_V_GEN)mkdir -p nanos6 ; \
	$(top_srcdir)/loader/scripts/generate_release_prototypes.sh $(MAX_REGION_DIMENSIONS) $(MULTIDIMENSIONAL_REGION_API_TYPES) > $@

multidim-release-fallbacks.h: $(top_srcdir)/loader/scripts/common.sh $(top_srcdir)/loader/scripts/generate_release_fallback_prototypes.sh Makefile
	$(AM_V_GEN)$(top_srcdir)/loader/scripts/generate_release_fallback_prototypes.sh $(MAX_REGION_DIMENSIONS) $(MULTIDIMENSIONAL_REGION_API_TYPES) > $@

multidim-release-fallbacks.c: $(top_srcdir)/loader/scripts/common.sh $(top_srcdir)/loader/scripts/generate_release_fallback_functions.sh Makefile multidim-release-fallbacks.h nanos6/multidimensional-release.h
	$(AM_V_GEN)$(top_srcdir)/loader/scripts/generate_release_fallback_functions.sh $(MAX_REGION_DIMENSIONS) $(MULTIDIMENSIONAL_REGION_API_TYPES) > $@

multidim-release-resolvers.c: $(top_srcdir)/loader/scripts/common.sh $(top_srcdir)/loader/scripts/generate_release_symbol_resolvers.sh Makefile multidim-release-fallbacks.h
	$(AM_V_GEN)$(top_srcdir)/loader/scripts/generate_release_symbol_resolvers.sh $(MAX_REGION_DIMENSIONS) $(MULTIDIMENSIONAL_REGION_API_TYPES) > $@

multidim-release-indirect-resolvers.c: $(top_srcdir)/loader/scripts/common.sh $(top_srcdir)/loader/scripts/generate_release_indirect_symbol_resolvers.sh Makefile multidim-release-fallbacks.h nanos6/multidimensional-release.h
	$(AM_V_GEN)$(top_srcdir)/loader/scripts/generate_release_indirect_symbol_resolvers.sh $(MAX_REGION_DIMENSIONS) $(MULTIDIMENSIONAL_REGION_API_TYPES) > $@


MultidimensionalAPIToLinear.cpp: $(top_srcdir)/loader/scripts/common.sh $(top_srcdir)/loader/scripts/generate_multidim_functions_for_linear_regions.sh Makefile
	$(AM_V_GEN)$(top_srcdir)/loader/scripts/generate_multidim_functions_for_linear_regions.sh $(MAX_REGION_DIMENSIONS) $(SUPPORTED_MULTIDIMENSIONAL_REGION_API_TYPES) > $@

MultidimensionalAPIToDiscrete.cpp: $(top_srcdir)/loader/scripts/common.sh $(top_srcdir)/loader/scripts/generate_multidim_functions_for_linear_regions.sh Makefile
	$(AM_V_GEN)$(top_srcdir)/loader/scripts/generate_multidim_functions_for_linear_regions.sh $(MAX_REGION_DIMENSIONS) $(SUPPORTED_MULTIDIMENSIONAL_REGION_API_TYPES) > $@


if HAVE_EXTRAE
   lib_LTLIBRARIES += libnanos6-extrae.la
endif

if HAVE_PAPI
   lib_LTLIBRARIES += libnanos6-stats-papi.la
endif




# Sources for debugging both the regular and the null runtime
universal_debug_sources = \
	src/lowlevel/FatalErrorHandler.cpp \
	src/lowlevel/FatalErrorHandler.hpp \
	src/system/RuntimeInfo.cpp \
	src/system/RuntimeInfo.hpp \
	src/system/RuntimeInfoAPI.cpp \
	src/system/debug/RuntimeVersionAPI.cpp \
	src/version/CodeVersionInfo.cpp \
	src/version/VersionInfo.hpp

nodist_universal_debug_sources = \
	src/version/CompilerVersionInfo.cpp


CLEANFILES += $(nodist_universal_debug_sources)
BUILT_SOURCES += $(nodist_universal_debug_sources)


nodist_common_sources = \
	$(nanos6_generated_headers)

common_sources = \
	src/executors/threads/CPU.cpp \
	src/executors/threads/CPU.hpp \
	src/executors/threads/CPUActivation.hpp \
	src/executors/threads/CPUManager.cpp \
	src/executors/threads/CPUManager.hpp \
	src/executors/threads/DefaultThreadManagerPolicy.cpp \
	src/executors/threads/DefaultThreadManagerPolicy.hpp \
	src/executors/threads/TaskFinalization.hpp \
	src/executors/threads/TaskFinalizationImplementation.hpp \
	src/executors/threads/ThreadManager.cpp \
	src/executors/threads/ThreadManager.hpp \
	src/executors/threads/ThreadManagerPolicy.cpp \
	src/executors/threads/ThreadManagerPolicy.hpp \
	src/executors/threads/ThreadManagerPolicyInterface.hpp \
	src/executors/threads/WorkerThread.cpp \
	src/executors/threads/WorkerThread.hpp \
	src/executors/threads/WorkerThreadImplementation.hpp \
	src/hardware/HardwareInfo.hpp \
	src/hardware/HardwareInfo.cpp \
	src/hardware/places/MemoryPlace.hpp \
	src/hardware/places/NUMAPlace.hpp \
	src/hardware/places/NUMAPlace.cpp \
	src/hardware/places/ComputePlace.hpp \
	src/hardware/places/ComputePlace.cpp \
	src/hardware/places/CPUPlace.hpp \
	src/hardware/places/CPUPlace.cpp \
	src/instrument/api/InstrumentAddTask.hpp \
	src/instrument/api/InstrumentDependenciesByAccess.hpp \
	src/instrument/api/InstrumentDependenciesByAccessLinks.hpp \
	src/instrument/api/InstrumentDependenciesByGroup.hpp \
	src/instrument/api/InstrumentComputePlaceManagement.hpp \
	src/instrument/api/InstrumentInitAndShutdown.hpp \
	src/instrument/api/InstrumentLeaderThread.hpp \
	src/instrument/api/InstrumentLogMessage.hpp \
	src/instrument/api/InstrumentTaskExecution.hpp \
	src/instrument/api/InstrumentTaskStatus.hpp \
	src/instrument/api/InstrumentTaskWait.hpp \
	src/instrument/api/InstrumentThreadManagement.hpp \
	src/instrument/api/InstrumentUserMutex.hpp \
	src/instrument/support/backtrace/BacktraceWalker.hpp \
	src/instrument/support/InstrumentHardwarePlaceManagement.hpp \
	src/instrument/support/InstrumentThreadLocalDataSupport.hpp \
	src/instrument/support/InstrumentThreadLocalDataSupportImplementation.hpp \
	src/lowlevel/BoostAssertionFailureHandler.cpp \
	src/lowlevel/ConditionVariable.hpp \
	src/lowlevel/EnvironmentVariable.hpp \
	src/lowlevel/PaddedSpinLock.hpp \
	src/lowlevel/PaddedTicketSpinLock.hpp \
	src/lowlevel/RWSpinLock.hpp \
	src/lowlevel/RWTicketSpinLock.hpp \
	src/lowlevel/SpinLock.hpp \
	src/lowlevel/SymbolResolver.hpp \
	src/lowlevel/TicketSpinLock.hpp \
	src/lowlevel/TokenizedEnvironmentVariable.hpp \
	src/lowlevel/apple/SpinLockImplementation.hpp \
	src/lowlevel/cxx/SpinLockImplementation.hpp \
	src/lowlevel/posix/SpinLockImplementation.hpp \
	src/lowlevel/threads/ExternalThread.cpp \
	src/lowlevel/threads/ExternalThread.hpp \
	src/lowlevel/threads/HelperThread.hpp \
	src/lowlevel/threads/KernelLevelThread.cpp \
	src/lowlevel/threads/KernelLevelThread.hpp \
	src/lowlevel/threads/posix/KernelLevelThread.hpp \
	src/memory/AddressSpace.hpp \
	src/scheduling/Scheduler.cpp \
	src/scheduling/Scheduler.hpp \
	src/scheduling/SchedulerGenerator.cpp \
	src/scheduling/SchedulerGenerator.hpp \
	src/scheduling/SchedulerInterface.cpp \
	src/scheduling/SchedulerInterface.hpp \
	src/scheduling/schedulers/DefaultScheduler.hpp \
	src/scheduling/schedulers/FIFOImmediateSuccessorWithMultiPollingScheduler.cpp \
	src/scheduling/schedulers/FIFOImmediateSuccessorWithMultiPollingScheduler.hpp \
	src/scheduling/schedulers/FIFOImmediateSuccessorWithPollingScheduler.cpp \
	src/scheduling/schedulers/FIFOImmediateSuccessorWithPollingScheduler.hpp \
	src/scheduling/schedulers/FIFOScheduler.cpp \
	src/scheduling/schedulers/FIFOScheduler.hpp \
	src/scheduling/schedulers/HostHierarchicalScheduler.cpp \
	src/scheduling/schedulers/HostHierarchicalScheduler.hpp \
	src/scheduling/schedulers/NUMAHierarchicalScheduler.cpp \
	src/scheduling/schedulers/NUMAHierarchicalScheduler.hpp \
	src/scheduling/schedulers/DeviceHierarchicalScheduler.cpp \
	src/scheduling/schedulers/DeviceHierarchicalScheduler.hpp \
	src/scheduling/schedulers/ImmediateSuccessorScheduler.cpp \
	src/scheduling/schedulers/ImmediateSuccessorScheduler.hpp \
	src/scheduling/schedulers/ImmediateSuccessorWithMultiPollingScheduler.cpp \
	src/scheduling/schedulers/ImmediateSuccessorWithMultiPollingScheduler.hpp \
	src/scheduling/schedulers/ImmediateSuccessorWithPollingScheduler.cpp \
	src/scheduling/schedulers/ImmediateSuccessorWithPollingScheduler.hpp \
	src/scheduling/schedulers/NaiveScheduler.cpp \
	src/scheduling/schedulers/NaiveScheduler.hpp \
	src/scheduling/schedulers/PriorityScheduler.cpp \
	src/scheduling/schedulers/PriorityScheduler.hpp \
	src/scheduling/TaskloopSchedulingPolicy.hpp \
	src/support/ConstPropagator.hpp \
	src/support/GlobalLock.cpp \
	src/support/GlobalLock.hpp \
	src/support/InlineDoublyLinkedList.hpp \
	src/support/InstrumentedThread.hpp \
	src/support/Objectified.hpp \
	src/support/StringComposer.hpp \
	src/support/StringLiteral.hpp \
	src/system/BlockingAPI.cpp \
	src/system/Bootstrap.cpp \
	src/system/If0Task.hpp \
	src/system/LeaderThread.cpp \
	src/system/LeaderThread.hpp \
	src/system/PollingAPI.cpp \
	src/system/PollingAPI.hpp \
	src/system/RuntimeInfoEssentials.cpp \
	src/system/RuntimeInfoEssentials.hpp \
	src/system/UtilsAPI.cpp \
	src/system/debug/DebugAPI.cpp \
	src/system/ompss/AddTask.cpp \
	src/system/ompss/Query.cpp \
	src/system/ompss/SpawnFunction.cpp \
	src/system/ompss/SpawnFunction.hpp \
	src/system/ompss/TaskBlocking.cpp \
	src/system/ompss/TaskBlocking.hpp \
	src/system/ompss/TaskWait.cpp \
	src/system/ompss/UserMutex.cpp \
	src/system/ompss/UserMutex.hpp \
	src/tasks/Task.hpp \
	src/tasks/TaskImplementation.hpp \
	src/tasks/TaskDebuggingInterface.hpp \
	src/tasks/Taskloop.cpp \
	src/tasks/Taskloop.hpp \
	src/tasks/TaskloopGenerator.hpp \
	src/tasks/TaskloopInfo.hpp \
	src/tasks/TaskloopLogic.hpp

common_dependency_sources = \
	src/dependencies/DataAccessType.hpp \
	src/dependencies/DataAccessBase.hpp \
	src/dependencies/MultidimensionalAPITraversal.hpp

discrete_dependency_sources = $(common_dependency_sources) \
	src/dependencies/discrete/CPUDependencyData.hpp \
	src/dependencies/discrete/DataAccess.hpp \
	src/dependencies/discrete/DataAccessImplementation.hpp \
	src/dependencies/discrete/DataAccessRegion.hpp \
	src/dependencies/discrete/DataAccessRegionIndexer.hpp \
	src/dependencies/discrete/DataAccessRegistration.hpp \
	src/dependencies/discrete/DataAccessSequence.hpp \
	src/dependencies/discrete/DataAccessSequenceImplementation.hpp \
	src/dependencies/discrete/DataAccessSequenceLinkingArtifacts.hpp \
	src/dependencies/discrete/DataAccessSequenceLinkingArtifactsImplementation.hpp \
	src/dependencies/discrete/DependencyDomain.hpp \
	src/dependencies/discrete/DependencySystem.hpp \
	src/dependencies/discrete/DiscreteAddressMap.hpp \
	src/dependencies/discrete/FixedAddressDataAccessMap.hpp \
	src/dependencies/discrete/MultidimensionalAPI.hpp \
	src/dependencies/discrete/Reductions.cpp \
	src/dependencies/discrete/RegisterDependencies.cpp \
	src/dependencies/discrete/RootDataAccessSequence.hpp \
	src/dependencies/discrete/RootDataAccessSequenceLinkingArtifacts.hpp \
	src/dependencies/discrete/RootDataAccessSequenceLinkingArtifactsImplementation.hpp \
	src/dependencies/discrete/TaskDataAccesses.hpp \
	src/dependencies/discrete/TaskDataAccessLinkingArtifacts.hpp \
	src/dependencies/discrete/TaskDataAccessLinkingArtifactsImplementation.hpp

common_linear_regions_dependency_sources = \
	src/dependencies/linear-regions/DataAccessRegion.hpp \
	src/dependencies/linear-regions/DataAccessRegionIndexer.hpp \
	src/dependencies/linear-regions/IntrusiveLinearRegionMap.hpp \
	src/dependencies/linear-regions/IntrusiveLinearRegionMapImplementation.hpp \
	src/dependencies/linear-regions/LinearRegionMap.hpp \
	src/dependencies/linear-regions/LinearRegionMapImplementation.hpp \
	src/dependencies/linear-regions/MultidimensionalAPI.hpp

linear_regions_unfragmented_dependency_sources = $(common_dependency_sources) $(common_linear_regions_dependency_sources) \
	src/dependencies/linear-regions-unfragmented/CPUDependencyData.hpp \
	src/dependencies/linear-regions-unfragmented/DataAccess.hpp \
	src/dependencies/linear-regions-unfragmented/DataAccessImplementation.hpp \
	src/dependencies/linear-regions-unfragmented/DataAccessRegistration.hpp \
	src/dependencies/linear-regions-unfragmented/DependencyDomain.hpp \
	src/dependencies/linear-regions-unfragmented/DependencySystem.hpp \
	src/dependencies/linear-regions-unfragmented/LinearRegionDataAccessMap.hpp \
	src/dependencies/linear-regions-unfragmented/LinearRegionDataAccessMapImplementation.hpp \
	src/dependencies/linear-regions-unfragmented/RegisterDependencies.cpp \
	src/dependencies/linear-regions-unfragmented/TaskDataAccesses.hpp \
	src/dependencies/linear-regions-unfragmented/TaskDataAccessLinkingArtifacts.hpp \
	src/dependencies/linear-regions-unfragmented/TaskDataAccessLinkingArtifactsImplementation.hpp

linear_regions_fragmented_dependency_sources = $(common_dependency_sources) $(common_linear_regions_dependency_sources) \
	src/dependencies/linear-regions-fragmented/BottomMapEntry.hpp \
	src/dependencies/linear-regions-fragmented/CPUDependencyData.hpp \
	src/dependencies/linear-regions-fragmented/DataAccess.hpp \
	src/dependencies/linear-regions-fragmented/DataAccessRegistration.hpp \
	src/dependencies/linear-regions-fragmented/DependencyDomain.hpp \
	src/dependencies/linear-regions-fragmented/DependencySystem.hpp \
	src/dependencies/linear-regions-fragmented/ReductionSpecific.hpp \
	src/dependencies/linear-regions-fragmented/Reductions.cpp \
	src/dependencies/linear-regions-fragmented/RegisterDependencies.cpp \
	src/dependencies/linear-regions-fragmented/ReleaseDirective.cpp \
	src/dependencies/linear-regions-fragmented/TaskDataAccesses.hpp \
	src/dependencies/linear-regions-fragmented/TaskDataAccessesImplementation.hpp \
	src/dependencies/linear-regions-fragmented/TaskDataAccessLinkingArtifacts.hpp \
	src/dependencies/linear-regions-fragmented/TaskDataAccessLinkingArtifactsImplementation.hpp

linear_regions_alternative_dependency_sources = $(common_dependency_sources) $(common_linear_regions_dependency_sources) \
	src/dependencies/linear-regions-alternative/BottomMapEntry.hpp \
	src/dependencies/linear-regions-alternative/CPUDependencyData.hpp \
	src/dependencies/linear-regions-alternative/DataAccess.hpp \
	src/dependencies/linear-regions-alternative/DataAccessRegistration.hpp \
	src/dependencies/linear-regions-alternative/DependencyDomain.hpp \
	src/dependencies/linear-regions-alternative/DependencySystem.hpp \
	src/dependencies/linear-regions-alternative/Reductions.cpp \
	src/dependencies/linear-regions-alternative/RegisterDependencies.cpp \
	src/dependencies/linear-regions-alternative/TaskDataAccesses.hpp \
	src/dependencies/linear-regions-alternative/TaskDataAccessesImplementation.hpp \
	src/dependencies/linear-regions-alternative/TaskDataAccessLinkingArtifacts.hpp \
	src/dependencies/linear-regions-alternative/TaskDataAccessLinkingArtifactsImplementation.hpp

multidimensional_adaptor_sources =

if DISCRETE_DEPENDENCIES
AM_CXXFLAGS += -I$(srcdir)/src/dependencies/discrete
common_sources += $(discrete_dependency_sources)
multidimensional_adaptor_sources += MultidimensionalAPIToDiscrete.cpp
SUPPORTED_MULTIDIMENSIONAL_REGION_API_TYPES += read write readwrite weak_read weak_write weak_readwrite concurrent reduction
endif

if LINEAR_REGION_UNFRAGMENTED_DEPENDENCIES
AM_CXXFLAGS += -I$(srcdir)/src/dependencies/linear-regions -I$(srcdir)/src/dependencies/linear-regions-unfragmented
common_sources += $(linear_regions_unfragmented_dependency_sources)
multidimensional_adaptor_sources += MultidimensionalAPIToLinear.cpp
SUPPORTED_MULTIDIMENSIONAL_REGION_API_TYPES += read write readwrite weak_read weak_write weak_readwrite
endif

if LINEAR_REGION_FRAGMENTED_DEPENDENCIES
AM_CXXFLAGS += -I$(srcdir)/src/dependencies/linear-regions -I$(srcdir)/src/dependencies/linear-regions-fragmented
common_sources += $(linear_regions_fragmented_dependency_sources)
multidimensional_adaptor_sources += MultidimensionalAPIToLinear.cpp
SUPPORTED_MULTIDIMENSIONAL_REGION_API_TYPES += read write readwrite weak_read weak_write weak_readwrite concurrent
endif
if LINEAR_REGION_ALTERNATIVE_DEPENDENCIES
AM_CXXFLAGS += -I$(srcdir)/src/dependencies/linear-regions -I$(srcdir)/src/dependencies/linear-regions-alternative
common_sources += $(linear_regions_alternative_dependency_sources)
multidimensional_adaptor_sources += MultidimensionalAPIToLinear.cpp
SUPPORTED_MULTIDIMENSIONAL_REGION_API_TYPES += read write readwrite weak_read weak_write weak_readwrite concurrent reduction
endif


introspection_sources = \
	src/instrument/support/introspection/CodeAddressInfoBase.cpp \
	src/instrument/support/introspection/CodeAddressInfoBase.hpp
introspection_cppflags =

if HAVE_ELFUTILS
introspection_sources += \
	src/instrument/support/introspection/ElfUtils/CodeAddressInfo.cpp \
	src/instrument/support/introspection/ElfUtils/CodeAddressInfo.hpp
introspection_cppflags += -I$(srcdir)/src/instrument/support/introspection/ElfUtils
else
introspection_sources += \
	src/instrument/support/introspection/Addr2Line/CodeAddressInfo.cpp \
	src/instrument/support/introspection/Addr2Line/CodeAddressInfo.hpp
introspection_cppflags += -I$(srcdir)/src/instrument/support/introspection/Addr2Line
endif

AM_CXXFLAGS += -I$(srcdir)/src/executors/threads/kernel-level
common_sources += \
	src/executors/threads/kernel-level/CPUThreadingModelData.cpp \
	src/executors/threads/kernel-level/CPUThreadingModelData.hpp \
	src/executors/threads/kernel-level/WorkerThreadBase.cpp \
	src/executors/threads/kernel-level/WorkerThreadBase.hpp


nodist_common_sources += \
	$(multidimensional_adaptor_sources)
BUILT_SOURCES += \
	$(multidimensional_adaptor_sources)
CLEANFILES += \
	$(multidimensional_adaptor_sources)


hardware_counters_sources = \
	src/performance/HardwareCounters.cpp \
	src/performance/HardwareCounters.hpp \
	src/performance/HardwareCountersThreadLocalData.hpp \
	src/performance/HardwareCountersThreadLocalDataImplementation.hpp

papi_hardware_counters_sources = \
	src/performance/PAPI/PAPIHardwareCounters.cpp \
	src/performance/PAPI/PAPIHardwareCounters.hpp \
	src/performance/PAPI/PAPIHardwareCountersThreadLocalData.hpp \
	src/performance/PAPI/PAPIHardwareCountersThreadLocalDataImplementation.hpp

no_hardware_counters_sources = \
	src/performance/no-HC/NoHardwareCounters.hpp \
	src/performance/no-HC/NoHardwareCountersThreadLocalData.hpp \
	src/performance/no-HC/NoHardwareCountersThreadLocalDataImplementation.hpp


noinstrument_sources = \
	src/instrument/null/InstrumentAddTask.hpp \
	src/instrument/null/InstrumentComputePlaceId.hpp \
	src/instrument/null/InstrumentComputePlaceManagement.hpp \
	src/instrument/null/InstrumentDataAccessId.hpp \
	src/instrument/null/InstrumentDependenciesByAccess.hpp \
	src/instrument/null/InstrumentDependenciesByAccessLinks.hpp \
	src/instrument/null/InstrumentDependenciesByGroup.hpp \
	src/instrument/null/InstrumentExternalThreadId.hpp \
	src/instrument/null/InstrumentExternalThreadLocalData.hpp \
	src/instrument/null/InstrumentInitAndShutdown.hpp \
	src/instrument/null/InstrumentInstrumentationContext.hpp \
	src/instrument/null/InstrumentLeaderThread.hpp \
	src/instrument/null/InstrumentLogMessage.hpp \
	src/instrument/null/InstrumentTaskExecution.hpp \
	src/instrument/null/InstrumentTaskId.hpp \
	src/instrument/null/InstrumentTaskStatus.hpp \
	src/instrument/null/InstrumentTaskWait.hpp \
	src/instrument/null/InstrumentThreadId.hpp \
	src/instrument/null/InstrumentThreadInstrumentationContext.hpp \
	src/instrument/null/InstrumentThreadInstrumentationContextImplementation.hpp \
	src/instrument/null/InstrumentThreadLocalData.hpp \
	src/instrument/null/InstrumentThreadManagement.hpp \
	src/instrument/null/InstrumentUserMutex.hpp

instrument_generic_ids_sources = \
	src/instrument/generic_ids/GenericIds.cpp \
	src/instrument/generic_ids/GenericIds.hpp \
	src/instrument/generic_ids/InstrumentExternalThreadId.hpp \
	src/instrument/generic_ids/InstrumentThreadId.hpp

instrument_graph_sources = \
	$(instrument_generic_ids_sources) \
	src/instrument/graph/Color.hpp \
	src/instrument/graph/ExecutionSteps.cpp \
	src/instrument/graph/ExecutionSteps.hpp \
	src/instrument/graph/GenerateEdges.cpp \
	src/instrument/graph/GenerateEdges.hpp \
	src/instrument/graph/InstrumentAddTask.cpp \
	src/instrument/graph/InstrumentAddTask.hpp \
	src/instrument/graph/InstrumentComputePlaceId.hpp \
	src/instrument/graph/InstrumentComputePlaceManagement.hpp \
	src/instrument/graph/InstrumentDataAccessId.hpp \
	src/instrument/graph/InstrumentDependenciesByAccess.hpp \
	src/instrument/graph/InstrumentDependenciesByAccessLinks.cpp \
	src/instrument/graph/InstrumentDependenciesByAccessLinks.hpp \
	src/instrument/graph/InstrumentDependenciesByGroup.hpp \
	src/instrument/graph/InstrumentExternalThreadId.hpp \
	src/instrument/graph/InstrumentExternalThreadLocalData.hpp \
	src/instrument/graph/InstrumentGraph.cpp \
	src/instrument/graph/InstrumentGraph.hpp \
	src/instrument/graph/InstrumentInitAndShutdown.cpp \
	src/instrument/graph/InstrumentInitAndShutdown.hpp \
	src/instrument/graph/InstrumentLeaderThread.hpp \
	src/instrument/graph/InstrumentLogMessage.hpp \
	src/instrument/graph/InstrumentTaskExecution.cpp \
	src/instrument/graph/InstrumentTaskExecution.hpp \
	src/instrument/graph/InstrumentTaskId.hpp \
	src/instrument/graph/InstrumentTaskStatus.hpp \
	src/instrument/graph/InstrumentTaskWait.cpp \
	src/instrument/graph/InstrumentTaskWait.hpp \
	src/instrument/graph/InstrumentThreadId.hpp \
	src/instrument/graph/InstrumentThreadLocalData.hpp \
	src/instrument/graph/InstrumentThreadManagement.hpp \
	src/instrument/graph/InstrumentUserMutex.cpp \
	src/instrument/graph/InstrumentUserMutex.hpp \
	src/instrument/graph/PathLength.cpp \
	src/instrument/graph/PathLength.hpp \
	src/instrument/graph/SortAccessGroups.cpp \
	src/instrument/graph/SortAccessGroups.hpp \
	src/instrument/support/InstrumentInstrumentationContext.hpp \
	src/instrument/support/InstrumentThreadInstrumentationContext.hpp \
	src/instrument/support/InstrumentThreadInstrumentationContextImplementation.hpp

instrument_profile_sources = \
	src/instrument/profile/Address.hpp \
	src/instrument/profile/InstrumentAddTask.hpp \
	src/instrument/profile/InstrumentComputePlaceId.hpp \
	src/instrument/profile/InstrumentComputePlaceManagement.hpp \
	src/instrument/profile/InstrumentDataAccessId.hpp \
	src/instrument/profile/InstrumentDependenciesByAccess.hpp \
	src/instrument/profile/InstrumentDependenciesByAccessLinks.hpp \
	src/instrument/profile/InstrumentDependenciesByGroup.hpp \
	src/instrument/profile/InstrumentExternalThreadId.hpp \
	src/instrument/profile/InstrumentExternalThreadLocalData.hpp \
	src/instrument/profile/InstrumentInitAndShutdown.cpp \
	src/instrument/profile/InstrumentInitAndShutdown.hpp \
	src/instrument/profile/InstrumentInstrumentationContext.hpp \
	src/instrument/profile/InstrumentLeaderThread.hpp \
	src/instrument/profile/InstrumentLogMessage.hpp \
	src/instrument/profile/InstrumentMalloc.cpp \
	src/instrument/profile/InstrumentProfile.cpp \
	src/instrument/profile/InstrumentProfile.hpp \
	src/instrument/profile/InstrumentTaskExecution.hpp \
	src/instrument/profile/InstrumentTaskId.hpp \
	src/instrument/profile/InstrumentTaskStatus.hpp \
	src/instrument/profile/InstrumentTaskWait.hpp \
	src/instrument/profile/InstrumentThreadId.hpp \
	src/instrument/profile/InstrumentThreadInstrumentationContext.hpp \
	src/instrument/profile/InstrumentThreadInstrumentationContextImplementation.hpp \
	src/instrument/profile/InstrumentThreadLocalData.hpp \
	src/instrument/profile/InstrumentThreadManagement.hpp \
	src/instrument/profile/InstrumentUserMutex.hpp \
	src/instrument/support/sampling/SigProf.cpp \
	src/instrument/support/sampling/SigProf.hpp \
	src/instrument/support/sampling/ThreadLocalData.hpp \
	$(introspection_sources)

instrument_stats_sources = \
	$(hardware_counters_sources) \
	src/instrument/stats/InstrumentAddTask.hpp \
	src/instrument/stats/InstrumentComputePlaceId.hpp \
	src/instrument/stats/InstrumentComputePlaceManagement.hpp \
	src/instrument/stats/InstrumentDataAccessId.hpp \
	src/instrument/stats/InstrumentDependenciesByAccess.hpp \
	src/instrument/stats/InstrumentDependenciesByAccessLinks.hpp \
	src/instrument/stats/InstrumentDependenciesByGroup.hpp \
	src/instrument/stats/InstrumentExternalThreadId.hpp \
	src/instrument/stats/InstrumentExternalThreadLocalData.hpp \
	src/instrument/stats/InstrumentInitAndShutdown.cpp \
	src/instrument/stats/InstrumentInitAndShutdown.hpp \
	src/instrument/stats/InstrumentLeaderThread.hpp \
	src/instrument/stats/InstrumentLogMessage.hpp \
	src/instrument/stats/InstrumentStats.cpp \
	src/instrument/stats/InstrumentStats.hpp \
	src/instrument/stats/InstrumentTaskExecution.hpp \
	src/instrument/stats/InstrumentTaskId.hpp \
	src/instrument/stats/InstrumentTaskStatus.hpp \
	src/instrument/stats/InstrumentTaskWait.hpp \
	src/instrument/stats/InstrumentThreadId.hpp \
	src/instrument/stats/InstrumentThreadLocalData.hpp \
	src/instrument/stats/InstrumentThreadManagement.hpp \
	src/instrument/stats/InstrumentUserMutex.hpp \
	src/instrument/stats/Timer.hpp \
	src/instrument/support/InstrumentInstrumentationContext.hpp \
	src/instrument/support/InstrumentStandardExternalThreadLocalData.hpp \
	src/instrument/support/InstrumentThreadInstrumentationContext.hpp \
	src/instrument/support/InstrumentThreadInstrumentationContextImplementation.hpp

instrument_verbose_sources = \
	$(instrument_generic_ids_sources) \
	src/instrument/verbose/InstrumentAddTask.cpp \
	src/instrument/verbose/InstrumentAddTask.hpp \
	src/instrument/verbose/InstrumentComputePlaceId.hpp \
	src/instrument/verbose/InstrumentComputePlaceManagement.hpp \
	src/instrument/verbose/InstrumentDataAccessId.hpp \
	src/instrument/verbose/InstrumentDependenciesByAccess.cpp \
	src/instrument/verbose/InstrumentDependenciesByAccess.hpp \
	src/instrument/verbose/InstrumentDependenciesByAccessLinks.cpp \
	src/instrument/verbose/InstrumentDependenciesByAccessLinks.hpp \
	src/instrument/verbose/InstrumentDependenciesByGroup.cpp \
	src/instrument/verbose/InstrumentDependenciesByGroup.hpp \
	src/instrument/verbose/InstrumentExternalThreadId.hpp \
	src/instrument/verbose/InstrumentExternalThreadLocalData.hpp \
	src/instrument/verbose/InstrumentInitAndShutdown.cpp \
	src/instrument/verbose/InstrumentInitAndShutdown.hpp \
	src/instrument/verbose/InstrumentLeaderThread.cpp \
	src/instrument/verbose/InstrumentLeaderThread.hpp \
	src/instrument/verbose/InstrumentLogMessage.hpp \
	src/instrument/verbose/InstrumentTaskExecution.cpp \
	src/instrument/verbose/InstrumentTaskExecution.hpp \
	src/instrument/verbose/InstrumentTaskId.hpp \
	src/instrument/verbose/InstrumentTaskStatus.cpp \
	src/instrument/verbose/InstrumentTaskStatus.hpp \
	src/instrument/verbose/InstrumentTaskWait.cpp \
	src/instrument/verbose/InstrumentTaskWait.hpp \
	src/instrument/verbose/InstrumentThreadId.hpp \
	src/instrument/verbose/InstrumentThreadLocalData.hpp \
	src/instrument/verbose/InstrumentThreadManagement.cpp \
	src/instrument/verbose/InstrumentThreadManagement.hpp \
	src/instrument/verbose/InstrumentUserMutex.cpp \
	src/instrument/verbose/InstrumentUserMutex.hpp \
	src/instrument/verbose/InstrumentVerbose.cpp \
	src/instrument/verbose/InstrumentVerbose.hpp \
	src/instrument/support/InstrumentInstrumentationContext.hpp \
	src/instrument/support/InstrumentStandardExternalThreadLocalData.hpp \
	src/instrument/support/InstrumentThreadInstrumentationContext.hpp \
	src/instrument/support/InstrumentThreadInstrumentationContextImplementation.hpp

instrument_extrae_sources = \
	$(instrument_generic_ids_sources) \
	src/instrument/extrae/InstrumentAddTask.hpp \
	src/instrument/extrae/InstrumentComputePlaceId.hpp \
	src/instrument/extrae/InstrumentComputePlaceManagement.hpp \
	src/instrument/extrae/InstrumentDataAccessId.hpp \
	src/instrument/extrae/InstrumentDependenciesByAccess.hpp \
	src/instrument/extrae/InstrumentDependenciesByAccessLinks.hpp \
	src/instrument/extrae/InstrumentDependenciesByGroup.hpp \
	src/instrument/extrae/InstrumentExternalThreadId.hpp \
	src/instrument/extrae/InstrumentExternalThreadLocalData.hpp \
	src/instrument/extrae/InstrumentExtrae.cpp \
	src/instrument/extrae/InstrumentExtrae.hpp \
	src/instrument/extrae/InstrumentInitAndShutdown.hpp \
	src/instrument/extrae/InstrumentLeaderThread.hpp \
	src/instrument/extrae/InstrumentLogMessage.hpp \
	src/instrument/extrae/InstrumentTaskExecution.hpp \
	src/instrument/extrae/InstrumentTaskId.hpp \
	src/instrument/extrae/InstrumentTaskStatus.hpp \
	src/instrument/extrae/InstrumentTaskWait.hpp \
	src/instrument/extrae/InstrumentThreadId.hpp \
	src/instrument/extrae/InstrumentThreadLocalData.hpp \
	src/instrument/extrae/InstrumentThreadManagement.hpp \
	src/instrument/extrae/InstrumentUserMutex.hpp \
	src/instrument/support/InstrumentInstrumentationContext.hpp \
	src/instrument/support/InstrumentThreadInstrumentationContext.hpp \
	src/instrument/support/InstrumentThreadInstrumentationContextImplementation.hpp


# null_common_sources = \
# 	src/null/NullDebug.cpp \
# 	src/null/NullAPI.cpp \
# 	src/null/NullSpawn.cpp
# 
# null_sources = \
# 	$(null_common_sources) \
# 	src/null/NullDynamicBlock.cpp
# 
# null_cb_sources = \
# 	$(null_common_sources) \
# 	src/null/NullConstantBlock.cpp
# 
# null_sb_sources = \
# 	$(null_common_sources) \
# 	src/null/NullStaticBlock.cpp

noinst_HEADERS = \
	tests/TestAnyProtocolProducer.hpp \
	tests/Timer.hpp

EXTRA_DIST += \
	tests/select-version.sh \
	tests/tap-driver.pl \
	tests/tap-driver.sh

common_libnanos6_cppflags = $(BOOST_CPPFLAGS) -DBOOST_ENABLE_ASSERT_DEBUG_HANDLER $(PTHREAD_CFLAGS) $(hwloc_CFLAGS) $(libnuma_CPPFLAGS)
common_libnanos6_ldflags = $(AM_LDFLAGS) $(BOOST_LDFLAGS) -version-info $(lib_current):$(lib_revision):$(lib_age) $(PTHREAD_CFLAGS) $(PTHREAD_LIBS) $(LDFLAGS_NOUNDEFINED) $(hwloc_LIBS) $(libnuma_LIBS)

libnanos6_optimized_la_CPPFLAGS = -DNDEBUG $(common_libnanos6_cppflags) -I$(srcdir)/src/instrument/null
libnanos6_optimized_la_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
libnanos6_optimized_la_SOURCES = $(common_sources) $(noinstrument_sources) $(universal_debug_sources)
libnanos6_optimized_la_LDFLAGS = $(common_libnanos6_ldflags)
nodist_libnanos6_optimized_la_SOURCES = $(nodist_common_sources) $(nodist_universal_debug_sources)

libnanos6_debug_la_CPPFLAGS = $(common_libnanos6_cppflags) -I$(srcdir)/src/instrument/null
libnanos6_debug_la_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
libnanos6_debug_la_SOURCES = $(common_sources) $(noinstrument_sources) $(universal_debug_sources) src/instrument/null/InstrumentMalloc.cpp
libnanos6_debug_la_LDFLAGS = $(common_libnanos6_ldflags) $(DLOPEN_LIBS)
nodist_libnanos6_debug_la_SOURCES = $(nodist_common_sources) $(nodist_universal_debug_sources)

libnanos6_graph_la_CPPFLAGS = -DNDEBUG $(common_libnanos6_cppflags) -I$(srcdir)/src/instrument/graph -I$(srcdir)/src/instrument/support
libnanos6_graph_la_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
libnanos6_graph_la_SOURCES = $(common_sources) $(instrument_graph_sources) $(universal_debug_sources)
libnanos6_graph_la_LDFLAGS = $(common_libnanos6_ldflags)
nodist_libnanos6_graph_la_SOURCES = $(nodist_common_sources) $(nodist_universal_debug_sources)

libnanos6_graph_debug_la_CPPFLAGS = $(common_libnanos6_cppflags) -I$(srcdir)/src/instrument/graph -I$(srcdir)/src/instrument/support
libnanos6_graph_debug_la_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
libnanos6_graph_debug_la_SOURCES = $(common_sources) $(instrument_graph_sources) $(universal_debug_sources)
libnanos6_graph_debug_la_LDFLAGS = $(common_libnanos6_ldflags)
nodist_libnanos6_graph_debug_la_SOURCES = $(nodist_common_sources) $(nodist_universal_debug_sources)

libnanos6_profile_la_CPPFLAGS = -DNDEBUG $(common_libnanos6_cppflags) -I$(srcdir)/src/instrument/profile $(elfutils_CFLAGS) $(introspection_cppflags)
libnanos6_profile_la_CXXFLAGS = $(PROFILE_CXXFLAGS) $(AM_CXXFLAGS) -g2
libnanos6_profile_la_SOURCES = $(common_sources) $(instrument_profile_sources) $(universal_debug_sources)
libnanos6_profile_la_LDFLAGS = $(common_libnanos6_ldflags) $(BACKTRACE_LIBS) $(CLOCK_LIBS) $(elfutils_LIBS) $(DLOPEN_LIBS)
nodist_libnanos6_profile_la_SOURCES = $(nodist_common_sources) $(nodist_universal_debug_sources)

libnanos6_stats_la_CPPFLAGS = -DNDEBUG $(common_libnanos6_cppflags) -I$(srcdir)/src/instrument/stats -I$(srcdir)/src/instrument/support
libnanos6_stats_la_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS) -g2
libnanos6_stats_la_SOURCES = $(common_sources) $(instrument_stats_sources) $(no_hardware_counters_sources) $(universal_debug_sources)
libnanos6_stats_la_LDFLAGS = $(common_libnanos6_ldflags) $(CLOCK_LIBS)
nodist_libnanos6_stats_la_SOURCES = $(nodist_common_sources) $(nodist_universal_debug_sources)

if HAVE_PAPI
libnanos6_stats_papi_la_CPPFLAGS = -DNDEBUG $(common_libnanos6_cppflags) -I$(srcdir)/src/instrument/stats -I$(srcdir)/src/instrument/support $(papi_CPPFLAGS) -DHAVE_PAPI
libnanos6_stats_papi_la_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS) -g2
libnanos6_stats_papi_la_SOURCES = $(common_sources) $(instrument_stats_sources) $(papi_sources) $(papi_hardware_counters_sources) $(universal_debug_sources)
libnanos6_stats_papi_la_LDFLAGS = $(common_libnanos6_ldflags) $(papi_LIBS) $(CLOCK_LIBS)
nodist_libnanos6_stats_papi_la_SOURCES = $(nodist_common_sources) $(nodist_universal_debug_sources)
endif

libnanos6_verbose_la_CPPFLAGS = -DNDEBUG $(common_libnanos6_cppflags) -I$(srcdir)/src/instrument/verbose -I$(srcdir)/src/instrument/support
libnanos6_verbose_la_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
libnanos6_verbose_la_SOURCES = $(common_sources) $(instrument_verbose_sources) $(universal_debug_sources)
libnanos6_verbose_la_LDFLAGS = $(common_libnanos6_ldflags) $(CLOCK_LIBS) $(ANDROID_LOG_LIBS)
nodist_libnanos6_verbose_la_SOURCES = $(nodist_common_sources) $(nodist_universal_debug_sources)

libnanos6_verbose_debug_la_CPPFLAGS = $(common_libnanos6_cppflags) -I$(srcdir)/src/instrument/verbose -I$(srcdir)/src/instrument/support
libnanos6_verbose_debug_la_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
libnanos6_verbose_debug_la_SOURCES = $(common_sources) $(instrument_verbose_sources) $(universal_debug_sources)
libnanos6_verbose_debug_la_LDFLAGS = $(common_libnanos6_ldflags) $(CLOCK_LIBS) $(ANDROID_LOG_LIBS)
nodist_libnanos6_verbose_debug_la_SOURCES = $(nodist_common_sources) $(nodist_universal_debug_sources)

# libnanos6_null_la_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
# libnanos6_null_la_SOURCES = $(null_sources) $(universal_debug_sources)
# libnanos6_null_la_LDFLAGS = $(common_libnanos6_ldflags)
# nodist_libnanos6_null_la_SOURCES = $(nodist_universal_debug_sources)
# 
# libnanos6_null_cb_la_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
# libnanos6_null_cb_la_SOURCES = $(null_cb_sources) $(universal_debug_sources)
# libnanos6_null_cb_la_LDFLAGS = $(common_libnanos6_ldflags)
# nodist_libnanos6_null_cb_la_SOURCES = $(nodist_universal_debug_sources)
# 
# libnanos6_null_sb_la_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
# libnanos6_null_sb_la_SOURCES = $(null_sb_sources) $(universal_debug_sources)
# libnanos6_null_sb_la_LDFLAGS = $(common_libnanos6_ldflags)
# nodist_libnanos6_null_sb_la_SOURCES = $(nodist_universal_debug_sources)

if HAVE_EXTRAE
libnanos6_extrae_la_CPPFLAGS = -DNDEBUG $(common_libnanos6_cppflags) -I$(srcdir)/src/instrument/extrae -I$(srcdir)/src/instrument/support @extrae_CPPFLAGS@
libnanos6_extrae_la_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
libnanos6_extrae_la_SOURCES = $(common_sources) $(instrument_extrae_sources) $(universal_debug_sources)
libnanos6_extrae_la_LDFLAGS = $(common_libnanos6_ldflags) @extrae_LDFLAGS@ @extrae_LIBS@ $(DLOPEN_LIBS)
nodist_libnanos6_extrae_la_SOURCES = $(nodist_common_sources) $(nodist_universal_debug_sources)
endif


uninstall-hook:
	@echo Removing deprecated headers
	for f in nanos6_rt_interface.h nanos6_debug_interface.h ; do \
		if test -f "$(DESTDIR)$(nanos6includedir)/$$f" ; then \
			rm "$(DESTDIR)$(nanos6includedir)/$$f" ; \
		fi ; \
	done
	@echo Removing deprecated runtime implementations
	for l in libnanos6-null.la libnanos6-null-cb.la libnanos6-null-sb.la ; do \
		if test -f "$(DESTDIR)$(libdir)/$$l" ; then \
			$(LIBTOOL) $(AM_LIBTOOLFLAGS) $(LIBTOOLFLAGS) --mode=uninstall rm -f "$(DESTDIR)$(libdir)/$$l" ; \
		fi ; \
	done


#
# Tests
#

unit_tests = inline-double-linked-list.debug.test inline-double-linked-list.test

unit_test_common_cxxflags = -I$(top_srcdir)/tests

inline_double_linked_list_debug_test_SOURCES = tests/unit/support/TestInlineDoublyLinkedList.cpp
inline_double_linked_list_debug_test_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS) $(unit_test_common_cxxflags)

inline_double_linked_list_test_SOURCES = tests/unit/support/TestInlineDoublyLinkedList.cpp
inline_double_linked_list_test_CPPFLAGS = -DNDEBUG
inline_double_linked_list_test_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS) $(unit_test_common_cxxflags)

check_PROGRAMS = $(unit_tests)
TESTS = $(unit_tests)
TEST_LOG_DRIVER = env AM_TAP_AWK='$(AWK)' $(SHELL) $(top_srcdir)/tests/tap-driver.sh





libtool: $(LIBTOOL_DEPS)
	$(SHELL) ./config.status libtool


if FROM_GIT_REPOSITORY
src/version/CodeVersionInfo.cpp: $(srcdir)/.git COPYING
	$(AM_V_GEN)mkdir -p src/version/ ; \
	echo '#include "src/version/VersionInfo.hpp"' > src/version/CodeVersionInfo.cpp ; \
	echo >> src/version/CodeVersionInfo.cpp ; \
	echo >> src/version/CodeVersionInfo.cpp ; \
	\
	echo -n 'char const * nanos6_version = "' >> src/version/CodeVersionInfo.cpp ; \
	ver=`$(GIT) --git-dir="$(srcdir)/.git" show --pretty=format:'%ci %h' -s HEAD` ; echo -n $$ver >> src/version/CodeVersionInfo.cpp ; \
	echo '";' >> src/version/CodeVersionInfo.cpp ; \
	\
	echo -n 'char const * nanos6_branch = "' >> src/version/CodeVersionInfo.cpp ; \
	branch=`$(GIT) --git-dir="$(srcdir)/.git" symbolic-ref HEAD | sed 's@refs/heads/@@'` ; echo -n $$branch >> src/version/CodeVersionInfo.cpp ; \
	echo '";' >> src/version/CodeVersionInfo.cpp ; \
	\
	echo >> src/version/CodeVersionInfo.cpp
endif
if EMBED_CODE_CHANGES
	$(AM_V_GEN2)echo -n 'char const * nanos6_patches = "' >> src/version/CodeVersionInfo.cpp ; \
	( \
		cd $(srcdir) ; \
		git diff ; \
		for i in `git ls-files --others --exclude-standard`; do \
			git diff --no-index /dev/null $$i ; \
		done \
	) | sed 's@\\@\\\\@g;s@"@\\"@g' | awk 'BEGIN {ORS="\\n";} {print $$0;}' >> src/version/CodeVersionInfo.cpp ; \
	echo '";' >> src/version/CodeVersionInfo.cpp
else
	$(AM_V_GEN2)echo 'char const * nanos6_patches = nullptr;' >> src/version/CodeVersionInfo.cpp
endif
	$(AM_V_GEN3)echo >> src/version/CodeVersionInfo.cpp ; \
	\
	echo 'char const * nanos6_copyright = "2015-2017 Barcelona Supercomputing Center (BSC)";' >> src/version/CodeVersionInfo.cpp ; \
	echo >> src/version/CodeVersionInfo.cpp ; \
	\
	echo 'char const * nanos6_license = "GPL3";' >> src/version/CodeVersionInfo.cpp ; \
	echo >> src/version/CodeVersionInfo.cpp ; \
	\
	echo -n 'char const * nanos6_full_license = "' >> src/version/CodeVersionInfo.cpp ; \
	cat $(top_srcdir)/COPYING | sed 's@\\@\\\\@g;s@"@\\"@g' | awk 'BEGIN {ORS="\\n";} {print $$0;}' >> src/version/CodeVersionInfo.cpp ; \
	echo '";' >> src/version/CodeVersionInfo.cpp ; \
	echo >> src/version/CodeVersionInfo.cpp


src/version/CompilerVersionInfo.cpp: Makefile
	$(AM_V_GEN)mkdir -p src/version/ ; \
	echo '#include "src/version/VersionInfo.hpp"' > src/version/CompilerVersionInfo.cpp ; \
	echo >> src/version/CompilerVersionInfo.cpp ; \
	echo >> src/version/CompilerVersionInfo.cpp ; \
	\
	echo 'char const * nanos6_compiler_version = "$(CXX_VERSION)";' >> src/version/CompilerVersionInfo.cpp ; \
	echo '#ifdef NDEBUG' >> src/version/CompilerVersionInfo.cpp ; \
	echo 'char const * nanos6_compiler_flags = "-DNDEBUG $(OPT_CXXFLAGS)";' >> src/version/CompilerVersionInfo.cpp ; \
	echo '#else' >> src/version/CompilerVersionInfo.cpp ; \
	echo 'char const * nanos6_compiler_flags = "$(DEBUG_CXXFLAGS)";' >> src/version/CompilerVersionInfo.cpp ; \
	echo '#endif' >> src/version/CompilerVersionInfo.cpp ; \
	echo >> src/version/CompilerVersionInfo.cpp


EXTRA_DIST += \
	COPYING \
	README.md

CLEANFILES += LICENSE.txt

LICENSE.txt: COPYING
	$(AM_V_GEN)cp $^ $@


doc_DATA = \
	LICENSE.txt


AM_V_CHECK = $(am__v_CHECK_@AM_V@)
am__v_CHECK_ = $(am__v_CHECK_@AM_DEFAULT_V@)
am__v_CHECK_0 = @echo "  CHECK   " $@;
am__v_CHECK_1 =


check-copyright: $(nodist_nanos6include_HEADERS)
	@for f in \
		$$(find $(top_srcdir) -name '*.[ch]pp' -or -name '*.[ch]') \
		$(nodist_nanos6include_HEADERS) \
		$(top_srcdir)/loader/scripts/*.sh \
	; do \
		if ! grep Copyright $$f > /dev/null ; then \
			echo Error: $$f is missing the copyright notice 1>&2 ; \
			fail=true; \
		fi ; \
	done ; \
	test -z $$fail


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
	$(DX_MAKEINDEX) refman.idx; \
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


MOSTLYCLEANFILES = $(DX_CLEANFILES)


build-tests-local: all $(check_PROGRAMS)

rpm: dist-bzip2
	$(MAKE) -C scripts rpm

deb: dist-bzip2
	$(MAKE) -C scripts deb
