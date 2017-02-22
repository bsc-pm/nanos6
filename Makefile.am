ACLOCAL_AMFLAGS = -I m4
AM_CFLAGS = -I$(srcdir)/api -I$(srcdir)/loader -I.
AM_CXXFLAGS = -I$(srcdir)/src -I$(builddir)/src -I$(srcdir)/src/instrument/support -I$(srcdir)/api -I.

SUBDIRS = . tests/directive_based


AM_CXXFLAGS += -I$(srcdir)/src/executors/threads/kernel-level


MAX_REGION_DIMENSIONS = 8
MULTIDIMENSIONAL_REGION_API_TYPES = read write readwrite weak_read weak_write weak_readwrite concurrent commutative
SUPPORTED_MULTIDIMENSIONAL_REGION_API_TYPES = 

nanos6includedir = $(includedir)/nanos6

nanos6include_HEADERS = \
	api/nanos6/nanos6_rt_interface.h \
	api/nanos6/nanos6_debug_interface.h \
	api/nanos6/blocking.h \
	api/nanos6/bootstrap.h \
	api/nanos6/constants.h \
	api/nanos6/debug.h \
	api/nanos6/dependencies.h \
	api/nanos6/final.h \
	api/nanos6/library-mode.h \
	api/nanos6/polling.h \
	api/nanos6/task-info-registration.h \
	api/nanos6/task-instantiation.h \
	api/nanos6/taskwait.h \
	api/nanos6/user-mutex.h

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
	libnanos6-verbose.la libnanos6-verbose-debug.la \
	libnanos6-null.la libnanos6-null-cb.la libnanos6-null-sb.la

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
	loader/intercept-main-common.c \
	loader/intercept-main-common.h \
	loader/main-wrapper.c \
	loader/main-wrapper.h


#
# Loader and symbol resolution
#
symbol_resolution =
symbol_resolution_header =
multidimensional_regions = 

if RESOLVE_SYMBOLS_USING_IFUNC
symbol_resolution_header += loader/symbol-resolver/resolve.h
symbol_resolution += \
	loader/symbol-resolver/blocking.c \
	loader/symbol-resolver/bootstrap.c \
	loader/symbol-resolver/cpu-control.c \
	loader/symbol-resolver/debugging.c \
	loader/symbol-resolver/dependencies.c \
	loader/symbol-resolver/final.c \
	loader/symbol-resolver/polling.c \
	loader/symbol-resolver/task-info-registration.c \
	loader/symbol-resolver/task-instantiation.c \
	loader/symbol-resolver/taskwait.c \
	loader/symbol-resolver/user-mutex.c \
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
	loader/indirect-symbols/polling.c \
	loader/indirect-symbols/task-info-registration.c \
	loader/indirect-symbols/task-instantiation.c \
	loader/indirect-symbols/taskwait.c \
	loader/indirect-symbols/user-mutex.c \
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
	loader/loader.c \
	loader/loader.h

libnanos6_la_LDFLAGS = $(PTHREAD_CFLAGS) $(PTHREAD_LIBS) $(DLOPEN_LIBS) $(LDFLAGS_NOUNDEFINED)


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

nanos6/multidimensional-dependencies.h: $(top_srcdir)/loader/scripts/common.sh $(top_srcdir)/loader/scripts/generate_regions_prototypes.sh Makefile
	$(AM_V_GEN)
	@mkdir -p nanos6
	@$(top_srcdir)/loader/scripts/generate_regions_prototypes.sh $(MAX_REGION_DIMENSIONS) $(MULTIDIMENSIONAL_REGION_API_TYPES) > $@

multidim-region-dependency-fallbacks.h: $(top_srcdir)/loader/scripts/common.sh $(top_srcdir)/loader/scripts/generate_regions_fallback_prototypes.sh Makefile
	$(AM_V_GEN)
	@$(top_srcdir)/loader/scripts/generate_regions_fallback_prototypes.sh $(MAX_REGION_DIMENSIONS) $(MULTIDIMENSIONAL_REGION_API_TYPES) > $@

multidim-region-dependency-fallbacks.c: $(top_srcdir)/loader/scripts/common.sh $(top_srcdir)/loader/scripts/generate_regions_fallback_functions.sh Makefile multidim-region-dependency-fallbacks.h nanos6/multidimensional-dependencies.h
	$(AM_V_GEN)
	@$(top_srcdir)/loader/scripts/generate_regions_fallback_functions.sh $(MAX_REGION_DIMENSIONS) $(MULTIDIMENSIONAL_REGION_API_TYPES) > $@

multidim-region-dependency-resolvers.c: $(top_srcdir)/loader/scripts/common.sh $(top_srcdir)/loader/scripts/generate_regions_symbol_resolvers.sh Makefile multidim-region-dependency-fallbacks.h
	$(AM_V_GEN)
	@$(top_srcdir)/loader/scripts/generate_regions_symbol_resolvers.sh $(MAX_REGION_DIMENSIONS) $(MULTIDIMENSIONAL_REGION_API_TYPES) > $@

multidim-region-dependency-indirect-resolvers.c: $(top_srcdir)/loader/scripts/common.sh $(top_srcdir)/loader/scripts/generate_regions_indirect_symbol_resolvers.sh Makefile multidim-region-dependency-fallbacks.h nanos6/multidimensional-dependencies.h
	$(AM_V_GEN)
	@$(top_srcdir)/loader/scripts/generate_regions_indirect_symbol_resolvers.sh $(MAX_REGION_DIMENSIONS) $(MULTIDIMENSIONAL_REGION_API_TYPES) > $@

nanos6/multidimensional-release.h: $(top_srcdir)/loader/scripts/common.sh $(top_srcdir)/loader/scripts/generate_release_prototypes.sh Makefile
	$(AM_V_GEN)
	@mkdir -p nanos6
	@$(top_srcdir)/loader/scripts/generate_release_prototypes.sh $(MAX_REGION_DIMENSIONS) $(MULTIDIMENSIONAL_REGION_API_TYPES) > $@

multidim-release-fallbacks.h: $(top_srcdir)/loader/scripts/common.sh $(top_srcdir)/loader/scripts/generate_release_fallback_prototypes.sh Makefile
	$(AM_V_GEN)
	@$(top_srcdir)/loader/scripts/generate_release_fallback_prototypes.sh $(MAX_REGION_DIMENSIONS) $(MULTIDIMENSIONAL_REGION_API_TYPES) > $@

multidim-release-fallbacks.c: $(top_srcdir)/loader/scripts/common.sh $(top_srcdir)/loader/scripts/generate_release_fallback_functions.sh Makefile multidim-release-fallbacks.h nanos6/multidimensional-release.h
	$(AM_V_GEN)
	@$(top_srcdir)/loader/scripts/generate_release_fallback_functions.sh $(MAX_REGION_DIMENSIONS) $(MULTIDIMENSIONAL_REGION_API_TYPES) > $@

multidim-release-resolvers.c: $(top_srcdir)/loader/scripts/common.sh $(top_srcdir)/loader/scripts/generate_release_symbol_resolvers.sh Makefile multidim-release-fallbacks.h
	$(AM_V_GEN)
	@$(top_srcdir)/loader/scripts/generate_release_symbol_resolvers.sh $(MAX_REGION_DIMENSIONS) $(MULTIDIMENSIONAL_REGION_API_TYPES) > $@

multidim-release-indirect-resolvers.c: $(top_srcdir)/loader/scripts/common.sh $(top_srcdir)/loader/scripts/generate_release_indirect_symbol_resolvers.sh Makefile multidim-release-fallbacks.h nanos6/multidimensional-release.h
	$(AM_V_GEN)
	@$(top_srcdir)/loader/scripts/generate_release_indirect_symbol_resolvers.sh $(MAX_REGION_DIMENSIONS) $(MULTIDIMENSIONAL_REGION_API_TYPES) > $@


MultidimensionalAPIToLinear.cpp: $(top_srcdir)/loader/scripts/common.sh $(top_srcdir)/loader/scripts/generate_multidim_functions_for_linear_regions.sh Makefile
	$(AM_V_GEN)
	@$(top_srcdir)/loader/scripts/generate_multidim_functions_for_linear_regions.sh $(MAX_REGION_DIMENSIONS) $(SUPPORTED_MULTIDIMENSIONAL_REGION_API_TYPES) > $@

MultidimensionalAPIToDiscrete.cpp: $(top_srcdir)/loader/scripts/common.sh $(top_srcdir)/loader/scripts/generate_multidim_functions_for_linear_regions.sh Makefile
	$(AM_V_GEN)
	@$(top_srcdir)/loader/scripts/generate_multidim_functions_for_linear_regions.sh $(MAX_REGION_DIMENSIONS) $(SUPPORTED_MULTIDIMENSIONAL_REGION_API_TYPES) > $@


if HAVE_EXTRAE
   lib_LTLIBRARIES += libnanos6-extrae.la
endif

if HAVE_PAPI
   lib_LTLIBRARIES += libnanos6-stats-papi.la
endif


# See info page of libtool "Updating version info"
lib_current = 0
lib_revision = 0
lib_age = 0


nodist_noinst_HEADERS = \
	src/version/VersionInfo.hpp


# Sources for debugging both the regular and the null runtime
universal_debug_sources = \
	src/system/debug/RuntimeVersionAPI.cpp

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
	src/executors/threads/ThreadManager.cpp \
	src/executors/threads/ThreadManager.hpp \
	src/executors/threads/ThreadManagerPolicy.cpp \
	src/executors/threads/ThreadManagerPolicy.hpp \
	src/executors/threads/ThreadManagerPolicyInterface.hpp \
	src/executors/threads/WorkerThread.cpp \
	src/executors/threads/WorkerThread.hpp \
	src/executors/threads/kernel-level/WorkerThreadBase.hpp \
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
	src/instrument/support/InstrumentInstrumentationContext.cpp \
	src/instrument/support/InstrumentInstrumentationContext.hpp \
	src/lowlevel/BoostAssertionFailureHandler.cpp \
	src/lowlevel/ConditionVariable.hpp \
	src/lowlevel/EnvironmentVariable.hpp \
	src/lowlevel/FatalErrorHandler.hpp \
	src/lowlevel/KernelLevelThread.cpp \
	src/lowlevel/KernelLevelThread.hpp \
	src/lowlevel/PaddedSpinLock.hpp \
	src/lowlevel/PaddedTicketSpinLock.hpp \
	src/lowlevel/RWSpinLock.hpp \
	src/lowlevel/RWTicketSpinLock.hpp \
	src/lowlevel/SpinLock.hpp \
	src/lowlevel/TicketSpinLock.hpp \
	src/memory/AddressSpace.hpp \
	src/lowlevel/TokenizedEnvironmentVariable.hpp \
	src/lowlevel/apple/SpinLockImplementation.hpp \
	src/lowlevel/cxx/SpinLockImplementation.hpp \
	src/lowlevel/posix/KernelLevelThread.hpp \
	src/lowlevel/posix/SpinLockImplementation.hpp \
	src/scheduling/DefaultScheduler.hpp \
	src/scheduling/FIFOImmediateSuccessorWithPollingScheduler.cpp \
	src/scheduling/FIFOImmediateSuccessorWithPollingScheduler.hpp \
	src/scheduling/FIFOScheduler.cpp \
	src/scheduling/FIFOScheduler.hpp \
	src/scheduling/hierarchy/HostHierarchicalScheduler.cpp \
	src/scheduling/hierarchy/HostHierarchicalScheduler.hpp \
	src/scheduling/hierarchy/NUMAHierarchicalScheduler.cpp \
	src/scheduling/hierarchy/NUMAHierarchicalScheduler.hpp \
	src/scheduling/ImmediateSuccessorScheduler.cpp \
	src/scheduling/ImmediateSuccessorScheduler.hpp \
	src/scheduling/ImmediateSuccessorWithPollingScheduler.cpp \
	src/scheduling/ImmediateSuccessorWithPollingScheduler.hpp \
	src/scheduling/NaiveScheduler.cpp \
	src/scheduling/NaiveScheduler.hpp \
	src/scheduling/PriorityScheduler.cpp \
	src/scheduling/PriorityScheduler.hpp \
	src/scheduling/Scheduler.cpp \
	src/scheduling/Scheduler.hpp \
	src/scheduling/SchedulerInterface.cpp \
	src/scheduling/SchedulerInterface.hpp \
	src/support/ConstPropagator.hpp \
	src/support/GlobalLock.cpp \
	src/support/GlobalLock.hpp \
	src/support/InlineDoublyLinkedList.hpp \
	src/support/InstrumentedThread.hpp \
	src/system/BlockingAPI.cpp \
	src/system/Bootstrap.cpp \
	src/system/If0Task.hpp \
	src/system/LeaderThread.cpp \
	src/system/LeaderThread.hpp \
	src/system/PollingAPI.cpp \
	src/system/PollingAPI.hpp \
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
	src/tasks/TaskDebuggingInterface.hpp

common_dependency_sources = \
	src/dependencies/DataAccessType.hpp \
	src/dependencies/DataAccessBase.hpp \
	src/dependencies/MultidimensionalAPITraversal.hpp

discrete_dependency_sources = $(common_dependency_sources) \
	src/dependencies/discrete/CPUDependencyData.hpp \
	src/dependencies/discrete/DataAccess.hpp \
	src/dependencies/discrete/DataAccessImplementation.hpp \
	src/dependencies/discrete/DataAccessRange.hpp \
	src/dependencies/discrete/DataAccessRangeIndexer.hpp \
	src/dependencies/discrete/DataAccessRegistration.hpp \
	src/dependencies/discrete/DataAccessSequence.hpp \
	src/dependencies/discrete/DataAccessSequenceImplementation.hpp \
	src/dependencies/discrete/DataAccessSequenceLinkingArtifacts.hpp \
	src/dependencies/discrete/DataAccessSequenceLinkingArtifactsImplementation.hpp \
	src/dependencies/discrete/DependencyDomain.hpp \
	src/dependencies/discrete/DiscreteAddressMap.hpp \
	src/dependencies/discrete/FixedAddressDataAccessMap.hpp \
	src/dependencies/discrete/MultidimensionalAPI.hpp \
	src/dependencies/discrete/RegisterDependencies.cpp \
	src/dependencies/discrete/RootDataAccessSequence.hpp \
	src/dependencies/discrete/RootDataAccessSequenceLinkingArtifacts.hpp \
	src/dependencies/discrete/RootDataAccessSequenceLinkingArtifactsImplementation.hpp \
	src/dependencies/discrete/TaskDataAccesses.hpp \
	src/dependencies/discrete/TaskDataAccessLinkingArtifacts.hpp \
	src/dependencies/discrete/TaskDataAccessLinkingArtifactsImplementation.hpp

common_linear_regions_dependency_sources = \
	src/dependencies/linear-regions/DataAccessRange.hpp \
	src/dependencies/linear-regions/DataAccessRangeIndexer.hpp \
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
	src/dependencies/linear-regions-fragmented/RegisterDependencies.cpp \
	src/dependencies/linear-regions-fragmented/ReleaseDirective.cpp \
	src/dependencies/linear-regions-fragmented/TaskDataAccesses.hpp \
	src/dependencies/linear-regions-fragmented/TaskDataAccessesImplementation.hpp \
	src/dependencies/linear-regions-fragmented/TaskDataAccessLinkingArtifacts.hpp \
	src/dependencies/linear-regions-fragmented/TaskDataAccessLinkingArtifactsImplementation.hpp


multidimensional_adaptor_sources =

if DISCRETE_DEPENDENCIES
AM_CXXFLAGS += -I$(srcdir)/src/dependencies/discrete
common_sources += $(discrete_dependency_sources)
multidimensional_adaptor_sources += MultidimensionalAPIToDiscrete.cpp
SUPPORTED_MULTIDIMENSIONAL_REGION_API_TYPES += read write readwrite weak_read weak_write weak_readwrite
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
SUPPORTED_MULTIDIMENSIONAL_REGION_API_TYPES += read write readwrite weak_read weak_write weak_readwrite
endif

nodist_common_sources += \
	$(multidimensional_adaptor_sources)
BUILT_SOURCES += \
	$(multidimensional_adaptor_sources)
CLEANFILES += \
	$(multidimensional_adaptor_sources)


hardware_counters_sources = \
	src/performance/HardwareCounters.hpp \
	src/performance/HardwareCounters.cpp

papi_hardware_counters_sources = \
	src/performance/PAPI/PAPIHardwareCounters.hpp \
	src/performance/PAPI/PAPIHardwareCounters.cpp

no_hardware_counters_sources = \
	src/performance/no-HC/NoHardwareCounters.hpp


noinstrument_sources = \
	src/instrument/null/InstrumentAddTask.hpp \
	src/instrument/null/InstrumentDataAccessId.hpp \
	src/instrument/null/InstrumentDependenciesByAccess.hpp \
	src/instrument/null/InstrumentDependenciesByAccessLinks.hpp \
	src/instrument/null/InstrumentDependenciesByGroup.hpp \
	src/instrument/null/InstrumentComputePlaceId.hpp \
	src/instrument/null/InstrumentComputePlaceManagement.hpp \
	src/instrument/null/InstrumentInitAndShutdown.hpp \
	src/instrument/null/InstrumentLeaderThread.hpp \
	src/instrument/null/InstrumentLogMessage.hpp \
	src/instrument/null/InstrumentTaskExecution.hpp \
	src/instrument/null/InstrumentTaskId.hpp \
	src/instrument/null/InstrumentTaskStatus.hpp \
	src/instrument/null/InstrumentTaskWait.hpp \
	src/instrument/null/InstrumentThreadId.hpp \
	src/instrument/null/InstrumentThreadManagement.hpp \
	src/instrument/null/InstrumentUserMutex.hpp

instrument_generic_ids_sources = \
	src/instrument/generic_ids/GenericIds.cpp \
	src/instrument/generic_ids/GenericIds.hpp \
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
	src/instrument/graph/InstrumentDataAccessId.hpp \
	src/instrument/graph/InstrumentDependenciesByAccess.hpp \
	src/instrument/graph/InstrumentDependenciesByAccessLinks.cpp \
	src/instrument/graph/InstrumentDependenciesByAccessLinks.hpp \
	src/instrument/graph/InstrumentDependenciesByGroup.hpp \
	src/instrument/graph/InstrumentGraph.cpp \
	src/instrument/graph/InstrumentGraph.hpp \
	src/instrument/graph/InstrumentComputePlaceId.hpp \
	src/instrument/graph/InstrumentComputePlaceManagement.hpp \
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
	src/instrument/graph/InstrumentThreadManagement.hpp \
	src/instrument/graph/InstrumentUserMutex.cpp \
	src/instrument/graph/InstrumentUserMutex.hpp \
	src/instrument/graph/PathLength.cpp \
	src/instrument/graph/PathLength.hpp \
	src/instrument/graph/SortAccessGroups.cpp \
	src/instrument/graph/SortAccessGroups.hpp

instrument_profile_sources = \
	src/instrument/profile/InstrumentAddTask.hpp \
	src/instrument/profile/InstrumentDataAccessId.hpp \
	src/instrument/profile/InstrumentDependenciesByAccess.hpp \
	src/instrument/profile/InstrumentDependenciesByAccessLinks.hpp \
	src/instrument/profile/InstrumentDependenciesByGroup.hpp \
	src/instrument/profile/InstrumentComputePlaceId.hpp \
	src/instrument/profile/InstrumentComputePlaceManagement.hpp \
	src/instrument/profile/InstrumentInitAndShutdown.cpp \
	src/instrument/profile/InstrumentInitAndShutdown.hpp \
	src/instrument/profile/InstrumentLeaderThread.hpp \
	src/instrument/profile/InstrumentLogMessage.hpp \
	src/instrument/profile/InstrumentProfile.cpp \
	src/instrument/profile/InstrumentProfile.hpp \
	src/instrument/profile/InstrumentTaskExecution.hpp \
	src/instrument/profile/InstrumentTaskId.hpp \
	src/instrument/profile/InstrumentTaskStatus.hpp \
	src/instrument/profile/InstrumentTaskWait.hpp \
	src/instrument/profile/InstrumentThreadId.hpp \
	src/instrument/profile/InstrumentThreadManagement.hpp \
	src/instrument/profile/InstrumentUserMutex.hpp

instrument_stats_sources = \
	$(hardware_counters_sources) \
	src/instrument/stats/InstrumentAddTask.hpp \
	src/instrument/stats/InstrumentDataAccessId.hpp \
	src/instrument/stats/InstrumentDependenciesByAccess.hpp \
	src/instrument/stats/InstrumentDependenciesByAccessLinks.hpp \
	src/instrument/stats/InstrumentDependenciesByGroup.hpp \
	src/instrument/stats/InstrumentComputePlaceId.hpp \
	src/instrument/stats/InstrumentComputePlaceManagement.hpp \
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
	src/instrument/stats/InstrumentThreadManagement.hpp \
	src/instrument/stats/InstrumentUserMutex.hpp \
	src/instrument/stats/Timer.hpp

instrument_verbose_sources = \
	$(instrument_generic_ids_sources) \
	src/instrument/verbose/InstrumentAddTask.cpp \
	src/instrument/verbose/InstrumentAddTask.hpp \
	src/instrument/verbose/InstrumentDataAccessId.hpp \
	src/instrument/verbose/InstrumentDependenciesByAccess.cpp \
	src/instrument/verbose/InstrumentDependenciesByAccess.hpp \
	src/instrument/verbose/InstrumentDependenciesByAccessLinks.cpp \
	src/instrument/verbose/InstrumentDependenciesByAccessLinks.hpp \
	src/instrument/verbose/InstrumentDependenciesByGroup.cpp \
	src/instrument/verbose/InstrumentDependenciesByGroup.hpp \
	src/instrument/verbose/InstrumentComputePlaceId.hpp \
	src/instrument/verbose/InstrumentComputePlaceManagement.hpp \
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
	src/instrument/verbose/InstrumentThreadManagement.cpp \
	src/instrument/verbose/InstrumentThreadManagement.hpp \
	src/instrument/verbose/InstrumentUserMutex.cpp \
	src/instrument/verbose/InstrumentUserMutex.hpp \
	src/instrument/verbose/InstrumentVerbose.cpp \
	src/instrument/verbose/InstrumentVerbose.hpp

instrument_extrae_sources = \
	$(instrument_generic_ids_sources) \
	src/instrument/extrae/InstrumentAddTask.hpp \
	src/instrument/extrae/InstrumentDataAccessId.hpp \
	src/instrument/extrae/InstrumentDependenciesByAccess.hpp \
	src/instrument/extrae/InstrumentDependenciesByAccessLinks.hpp \
	src/instrument/extrae/InstrumentDependenciesByGroup.hpp \
	src/instrument/extrae/InstrumentExtrae.cpp \
	src/instrument/extrae/InstrumentExtrae.hpp \
	src/instrument/extrae/InstrumentComputePlaceId.hpp \
	src/instrument/extrae/InstrumentComputePlaceManagement.hpp \
	src/instrument/extrae/InstrumentInitAndShutdown.hpp \
	src/instrument/extrae/InstrumentLeaderThread.hpp \
	src/instrument/extrae/InstrumentLogMessage.hpp \
	src/instrument/extrae/InstrumentTaskExecution.hpp \
	src/instrument/extrae/InstrumentTaskId.hpp \
	src/instrument/extrae/InstrumentTaskStatus.hpp \
	src/instrument/extrae/InstrumentTaskWait.hpp \
	src/instrument/extrae/InstrumentThreadId.hpp \
	src/instrument/extrae/InstrumentThreadManagement.hpp \
	src/instrument/extrae/InstrumentUserMutex.hpp


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
	tests/TestAnyProtocolProducer.hpp \
	tests/Timer.hpp

EXTRA_DIST += \
	tests/select-version.sh \
	tests/tap-driver.pl \
	tests/tap-driver.sh

common_libnanos6_cppflags = $(BOOST_CPPFLAGS) -DBOOST_ENABLE_ASSERT_DEBUG_HANDLER $(PTHREAD_CFLAGS) $(hwloc_CFLAGS) $(libnuma_CPPFLAGS)
common_libnanos6_ldflags = $(BOOST_LDFLAGS) -version-info $(lib_current):$(lib_revision):$(lib_age) $(PTHREAD_CFLAGS) $(PTHREAD_LIBS) $(LDFLAGS_NOUNDEFINED) $(hwloc_LIBS) $(libnuma_LIBS)

libnanos6_optimized_la_CPPFLAGS = -DNDEBUG $(common_libnanos6_cppflags) -I$(srcdir)/src/instrument/null
libnanos6_optimized_la_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
libnanos6_optimized_la_SOURCES = $(common_sources) $(noinstrument_sources) $(universal_debug_sources)
libnanos6_optimized_la_LDFLAGS = $(common_libnanos6_ldflags)
nodist_libnanos6_optimized_la_SOURCES = $(nodist_common_sources)

libnanos6_debug_la_CPPFLAGS = $(common_libnanos6_cppflags) -I$(srcdir)/src/instrument/null
libnanos6_debug_la_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
libnanos6_debug_la_SOURCES = $(common_sources) $(noinstrument_sources) $(universal_debug_sources)
libnanos6_debug_la_LDFLAGS = $(common_libnanos6_ldflags)
nodist_libnanos6_debug_la_SOURCES = $(nodist_common_sources)

libnanos6_graph_la_CPPFLAGS = -DNDEBUG $(common_libnanos6_cppflags) -I$(srcdir)/src/instrument/graph
libnanos6_graph_la_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
libnanos6_graph_la_SOURCES = $(common_sources) $(instrument_graph_sources) $(universal_debug_sources)
libnanos6_graph_la_LDFLAGS = $(common_libnanos6_ldflags)
nodist_libnanos6_graph_la_SOURCES = $(nodist_common_sources)

libnanos6_graph_debug_la_CPPFLAGS = $(common_libnanos6_cppflags) -I$(srcdir)/src/instrument/graph
libnanos6_graph_debug_la_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
libnanos6_graph_debug_la_SOURCES = $(common_sources) $(instrument_graph_sources) $(universal_debug_sources)
libnanos6_graph_debug_la_LDFLAGS = $(common_libnanos6_ldflags)
nodist_libnanos6_graph_debug_la_SOURCES = $(nodist_common_sources)

libnanos6_profile_la_CPPFLAGS = -DNDEBUG $(common_libnanos6_cppflags) -I$(srcdir)/src/instrument/profile $(elfutils_CFLAGS)
libnanos6_profile_la_CXXFLAGS = $(PROFILE_CXXFLAGS) $(AM_CXXFLAGS) -g2
libnanos6_profile_la_SOURCES = $(common_sources) $(instrument_profile_sources) $(universal_debug_sources)
libnanos6_profile_la_LDFLAGS = $(common_libnanos6_ldflags) $(BACKTRACE_LIBS) $(CLOCK_LIBS) $(elfutils_LIBS)
nodist_libnanos6_profile_la_SOURCES = $(nodist_common_sources)

libnanos6_stats_la_CPPFLAGS = -DNDEBUG $(common_libnanos6_cppflags) -I$(srcdir)/src/instrument/stats
libnanos6_stats_la_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS) -g2
libnanos6_stats_la_SOURCES = $(common_sources) $(instrument_stats_sources) $(no_hardware_counters_sources) $(universal_debug_sources)
libnanos6_stats_la_LDFLAGS = $(common_libnanos6_ldflags) $(CLOCK_LIBS)
nodist_libnanos6_stats_la_SOURCES = $(nodist_common_sources)

if HAVE_PAPI
libnanos6_stats_papi_la_CPPFLAGS = -DNDEBUG $(common_libnanos6_cppflags) -I$(srcdir)/src/instrument/stats $(papi_CPPFLAGS) -DHAVE_PAPI
libnanos6_stats_papi_la_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS) -g2
libnanos6_stats_papi_la_SOURCES = $(common_sources) $(instrument_stats_sources) $(papi_sources) $(papi_hardware_counters_sources) $(universal_debug_sources)
libnanos6_stats_papi_la_LDFLAGS = $(common_libnanos6_ldflags) $(papi_LIBS) $(CLOCK_LIBS)
nodist_libnanos6_stats_papi_la_SOURCES = $(nodist_common_sources)
endif

libnanos6_verbose_la_CPPFLAGS = -DNDEBUG $(common_libnanos6_cppflags) -I$(srcdir)/src/instrument/verbose
libnanos6_verbose_la_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
libnanos6_verbose_la_SOURCES = $(common_sources) $(instrument_verbose_sources) $(universal_debug_sources)
libnanos6_verbose_la_LDFLAGS = $(common_libnanos6_ldflags) $(CLOCK_LIBS) $(ANDROID_LOG_LIBS)
nodist_libnanos6_verbose_la_SOURCES = $(nodist_common_sources)

libnanos6_verbose_debug_la_CPPFLAGS = $(common_libnanos6_cppflags) -I$(srcdir)/src/instrument/verbose
libnanos6_verbose_debug_la_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
libnanos6_verbose_debug_la_SOURCES = $(common_sources) $(instrument_verbose_sources) $(universal_debug_sources)
libnanos6_verbose_debug_la_LDFLAGS = $(common_libnanos6_ldflags) $(CLOCK_LIBS) $(ANDROID_LOG_LIBS)
nodist_libnanos6_verbose_debug_la_SOURCES = $(nodist_common_sources)

libnanos6_null_la_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
libnanos6_null_la_SOURCES = $(null_sources) $(universal_debug_sources)
libnanos6_null_la_LDFLAGS = $(common_libnanos6_ldflags)

libnanos6_null_cb_la_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
libnanos6_null_cb_la_SOURCES = $(null_cb_sources) $(universal_debug_sources)
libnanos6_null_cb_la_LDFLAGS = $(common_libnanos6_ldflags)

libnanos6_null_sb_la_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
libnanos6_null_sb_la_SOURCES = $(null_sb_sources) $(universal_debug_sources)
libnanos6_null_sb_la_LDFLAGS = $(common_libnanos6_ldflags)

if HAVE_EXTRAE
libnanos6_extrae_la_CPPFLAGS = -DNDEBUG $(common_libnanos6_cppflags) -I$(srcdir)/src/instrument/extrae @extrae_CPPFLAGS@
libnanos6_extrae_la_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
libnanos6_extrae_la_SOURCES = $(common_sources) $(instrument_extrae_sources) $(universal_debug_sources)
libnanos6_extrae_la_LDFLAGS = @extrae_LDFLAGS@ @extrae_LIBS@ $(common_libnanos6_ldflags)
nodist_libnanos6_extrae_la_SOURCES = $(nodist_common_sources)
endif



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

