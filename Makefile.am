#	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
#	
#	Copyright (C) 2015-2019 Barcelona Supercomputing Center (BSC)

# A workaround for http://debbugs.gnu.org/cgi/bugreport.cgi?bug=18744
am__is_gnu_make = { \
  if test -z '$(MAKELEVEL)'; then \
    false; \
  elif test -n '$(MAKE_HOST)'; then \
    true; \
  elif test -n '$(MAKE_VERSION)' && test -n '$(CURDIR)'; then \
    true; \
  else \
    false; \
  fi; \
}


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
MULTIDIMENSIONAL_REGION_API_TYPES = read write readwrite weak_read weak_write weak_readwrite concurrent commutative weak_commutative reduction weak_reduction
SUPPORTED_MULTIDIMENSIONAL_REGION_API_TYPES = 

nanos6includedir = $(includedir)/nanos6

nanos6include_HEADERS = \
	api/nanos6/api-check.h \
	api/nanos6/blocking.h \
	api/nanos6/bootstrap.h \
	api/nanos6/cluster.h \
	api/nanos6/constants.h \
	api/nanos6/cuda_device.h \
	api/nanos6/debug.h \
	api/nanos6/devices.h \
	api/nanos6/events.h \
	api/nanos6/final.h \
	api/nanos6/library-mode.h \
	api/nanos6/major.h \
	api/nanos6/monitoring.h \
	api/nanos6/polling.h \
	api/nanos6/reductions.h \
	api/nanos6/runtime-info.h \
	api/nanos6/task-info-registration.h \
	api/nanos6/task-instantiation.h \
	api/nanos6/taskloop.h \
	api/nanos6/taskwait.h \
	api/nanos6/user-mutex.h

nodist_nanos6include_HEADERS = \
	nanos6/multidimensional-dependencies.h \
	nanos6/multidimensional-release.h 

include_HEADERS = \
	api/nanos6.h


lib_LTLIBRARIES = \
	libnanos6.la \
	libnanos6-optimized.la \
	libnanos6-debug.la \
	libnanos6-extrae.la \
	libnanos6-graph.la \
	libnanos6-graph-debug.la \
	libnanos6-profile.la \
	libnanos6-stats.la \
	libnanos6-stats-papi.la \
	libnanos6-verbose.la \
	libnanos6-verbose-debug.la


noinst_LIBRARIES = libnanos6-main-wrapper.a libnanos6-library-mode.a
lib_OBJECTS = nanos6-main-wrapper.o nanos6-library-mode.o


AM_V_LD = $(am__v_LD_@AM_V@)
am__v_LD_ = $(am__v_LD_@AM_DEFAULT_V@)
am__v_LD_0 = @echo "  LD      " $@;
am__v_LD_1 = 



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
	loader/api-versions.c \
	loader/api-versions.h \
	loader/device_strings.c \
	loader/error.h \
	loader/function-interception.h \
	loader/intercept-main-common.c \
	loader/intercept-main-common.h \
	loader/main-wrapper.c \
	loader/main-wrapper.h

libnanos6_main_wrapper_a_CPPFLAGS = $(CUDA_CFLAGS) $(MPI_CXXFLAGS) $(MCMODEL_FLAGS)


nanos6-main-wrapper.o: libnanos6-main-wrapper.a
	$(AM_V_LD)$(LD) -r --whole-archive libnanos6-main-wrapper.a -o nanos6-main-wrapper.o


libnanos6_library_mode_a_SOURCES = \
	$(library_interception) \
	loader/api-versions.c \
	loader/api-versions.h \
	loader/device_strings.c \
	loader/library-mode-init.c

libnanos6_library_mode_a_CPPFLAGS = $(CUDA_CFLAGS) $(MPI_CXXFLAGS) $(MCMODEL_FLAGS)


nanos6-library-mode.o: libnanos6-library-mode.a
	$(AM_V_LD)$(LD) -r --whole-archive libnanos6-library-mode.a -o nanos6-library-mode.o


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
	loader/symbol-resolver/api-check.c \
	loader/symbol-resolver/blocking.c \
	loader/symbol-resolver/bootstrap.c \
	loader/symbol-resolver/cluster.c \
	loader/symbol-resolver/cpu-control.c \
	loader/symbol-resolver/debugging.c \
	loader/symbol-resolver/dependencies.c \
	loader/symbol-resolver/events.c \
	loader/symbol-resolver/final.c \
	loader/symbol-resolver/monitoring.c \
	loader/symbol-resolver/polling.c \
	loader/symbol-resolver/reductions.c \
	loader/symbol-resolver/runtime-info.c \
	loader/symbol-resolver/task-info-registration.c \
	loader/symbol-resolver/task-instantiation.c \
	loader/symbol-resolver/taskloop.c \
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
	loader/indirect-symbols/api-check.c \
	loader/indirect-symbols/blocking.c \
	loader/indirect-symbols/bootstrap.c \
	loader/indirect-symbols/cluster.c \
	loader/indirect-symbols/cpu-control.c \
	loader/indirect-symbols/debugging.c \
	loader/indirect-symbols/dependencies.c \
	loader/indirect-symbols/events.c \
	loader/indirect-symbols/final.c \
	loader/indirect-symbols/malloc.c \
	loader/indirect-symbols/monitoring.c \
	loader/indirect-symbols/polling.c \
	loader/indirect-symbols/reductions.c \
	loader/indirect-symbols/runtime-info.c \
	loader/indirect-symbols/task-info-registration.c \
	loader/indirect-symbols/task-instantiation.c \
	loader/indirect-symbols/taskloop.c \
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
	loader/error.h \
	loader/loader.c \
	loader/loader.h

libnanos6_la_CPPFLAGS = -DSONAME_MAJOR=\"$(lib_major)\" -DSONAME_SUFFIX=\"$(lib_suffix)\" $(CUDA_CFLAGS) $(MPI_CXXFLAGS)
libnanos6_la_LDFLAGS = $(AM_LDFLAGS) $(PTHREAD_CFLAGS) $(PTHREAD_LIBS) $(DLOPEN_LIBS) $(LDFLAGS_NOUNDEFINED)


nanos6_generated_headers = \
	nanos6/multidimensional-dependencies.h \
	nanos6/multidimensional-release.h
	
nodist_libnanos6_la_SOURCES = \
	$(nanos6_generated_headers) \
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
	loader/scripts/generate_regions_indirect_symbol_resolvers.sh \
	loader/scripts/generate_regions_prototypes.sh \
	loader/scripts/generate_regions_symbol_resolvers.sh \
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

multidim-region-dependency-resolvers.c: $(top_srcdir)/loader/scripts/common.sh $(top_srcdir)/loader/scripts/generate_regions_symbol_resolvers.sh Makefile
	$(AM_V_GEN)$(top_srcdir)/loader/scripts/generate_regions_symbol_resolvers.sh $(MAX_REGION_DIMENSIONS) $(MULTIDIMENSIONAL_REGION_API_TYPES) > $@

multidim-region-dependency-indirect-resolvers.c: $(top_srcdir)/loader/scripts/common.sh $(top_srcdir)/loader/scripts/generate_regions_indirect_symbol_resolvers.sh Makefile nanos6/multidimensional-dependencies.h
	$(AM_V_GEN)$(top_srcdir)/loader/scripts/generate_regions_indirect_symbol_resolvers.sh $(MAX_REGION_DIMENSIONS) $(MULTIDIMENSIONAL_REGION_API_TYPES) > $@

nanos6/multidimensional-release.h: $(top_srcdir)/loader/scripts/common.sh $(top_srcdir)/loader/scripts/generate_release_prototypes.sh Makefile
	$(AM_V_GEN)mkdir -p nanos6 ; \
	$(top_srcdir)/loader/scripts/generate_release_prototypes.sh $(MAX_REGION_DIMENSIONS) $(MULTIDIMENSIONAL_REGION_API_TYPES) > $@

multidim-release-resolvers.c: $(top_srcdir)/loader/scripts/common.sh $(top_srcdir)/loader/scripts/generate_release_symbol_resolvers.sh Makefile
	$(AM_V_GEN)$(top_srcdir)/loader/scripts/generate_release_symbol_resolvers.sh $(MAX_REGION_DIMENSIONS) $(MULTIDIMENSIONAL_REGION_API_TYPES) > $@

multidim-release-indirect-resolvers.c: $(top_srcdir)/loader/scripts/common.sh $(top_srcdir)/loader/scripts/generate_release_indirect_symbol_resolvers.sh Makefile nanos6/multidimensional-release.h
	$(AM_V_GEN)$(top_srcdir)/loader/scripts/generate_release_indirect_symbol_resolvers.sh $(MAX_REGION_DIMENSIONS) $(MULTIDIMENSIONAL_REGION_API_TYPES) > $@


MultidimensionalAPIToLinear.cpp: $(top_srcdir)/loader/scripts/common.sh $(top_srcdir)/loader/scripts/generate_multidim_functions_for_linear_regions.sh Makefile
	$(AM_V_GEN)$(top_srcdir)/loader/scripts/generate_multidim_functions_for_linear_regions.sh $(MAX_REGION_DIMENSIONS) $(SUPPORTED_MULTIDIMENSIONAL_REGION_API_TYPES) > $@

MultidimensionalAPIToDiscrete.cpp: $(top_srcdir)/loader/scripts/common.sh $(top_srcdir)/loader/scripts/generate_multidim_functions_for_linear_regions.sh Makefile
	$(AM_V_GEN)$(top_srcdir)/loader/scripts/generate_multidim_functions_for_linear_regions.sh $(MAX_REGION_DIMENSIONS) $(SUPPORTED_MULTIDIMENSIONAL_REGION_API_TYPES) > $@




# Sources for debugging
universal_debug_sources = \
	src/lowlevel/FatalErrorHandler.cpp \
	src/system/RuntimeInfo.cpp \
	src/system/RuntimeInfoAPI.cpp \
	src/system/debug/RuntimeVersionAPI.cpp \
	src/version/CodeVersionInfo.cpp

nodist_universal_debug_sources = \
	src/version/CompilerVersionInfo.cpp


CLEANFILES += $(nodist_universal_debug_sources)
BUILT_SOURCES += $(nodist_universal_debug_sources)


nodist_common_sources = \
	$(nanos6_generated_headers)

common_sources = \
	src/executors/threads/CPU.cpp \
	src/executors/threads/CPUManager.cpp \
	src/executors/threads/ThreadManager.cpp \
	src/executors/threads/WorkerThread.cpp \
	src/hardware/HardwareInfo.cpp \
	src/hardware/hwinfo/HostInfo.cpp \
	src/hardware/places/CPUPlace.cpp \
	src/hardware/places/ComputePlace.cpp \
	src/hardware/places/NUMAPlace.cpp \
	src/lowlevel/BoostAssertionFailureHandler.cpp \
	src/lowlevel/SymbolResolver.cpp \
	src/lowlevel/threads/ExternalThread.cpp \
	src/lowlevel/threads/ExternalThreadGroup.cpp \
	src/lowlevel/threads/KernelLevelThread.cpp \
	src/memory/directory/Directory.cpp \
	src/memory/directory/HomeNodeMap.cpp \
	src/scheduling/Scheduler.cpp \
	src/scheduling/SchedulerGenerator.cpp \
	src/scheduling/SchedulerInterface.cpp \
	src/scheduling/schedulers/DeviceHierarchicalScheduler.cpp \
	src/scheduling/schedulers/FIFOImmediateSuccessorWithPollingScheduler.cpp \
	src/scheduling/schedulers/FIFOScheduler.cpp \
	src/scheduling/schedulers/HostHierarchicalScheduler.cpp \
	src/scheduling/schedulers/ImmediateSuccessorScheduler.cpp \
	src/scheduling/schedulers/ImmediateSuccessorWithPollingScheduler.cpp \
	src/scheduling/schedulers/NUMAHierarchicalScheduler.cpp \
	src/scheduling/schedulers/NaiveScheduler.cpp \
	src/scheduling/schedulers/NoSleepPriorityScheduler.cpp \
	src/scheduling/schedulers/PriorityScheduler.cpp \
	src/scheduling/schedulers/PriorityScheduler1.cpp \
	src/scheduling/schedulers/cluster/ClusterLocalityScheduler.cpp \
	src/scheduling/schedulers/cluster/ClusterRandomScheduler.cpp \
	src/support/GlobalLock.cpp \
	src/system/APICheck.cpp \
	src/system/BlockingAPI.cpp \
	src/system/Bootstrap.cpp \
	src/system/ClusterAPI.cpp \
	src/system/EventsAPI.cpp \
	src/system/LeaderThread.cpp \
	src/system/MonitoringAPI.cpp \
	src/system/PollingAPI.cpp \
	src/system/RuntimeInfoEssentials.cpp \
	src/system/debug/DebugAPI.cpp \
	src/system/ompss/AddTask.cpp \
	src/system/ompss/Query.cpp \
	src/system/ompss/SpawnFunction.cpp \
	src/system/ompss/TaskBlocking.cpp \
	src/system/ompss/TaskLoop.cpp \
	src/system/ompss/TaskWait.cpp \
	src/system/ompss/UserMutex.cpp \
	src/tasks/Taskloop.cpp

cuda_sources = \
	src/executors/cuda/CUDAPollingService.cpp \
	src/hardware/cuda/CUDAInfo.cpp \
	src/hardware/cuda/compute/CUDAComputePlace.cpp \
	src/hardware/cuda/memory/CUDAMemoryPlace.cpp \
	src/scheduling/schedulers/cuda/CUDANaiveScheduler.cpp 

if USE_CUDA
common_sources += $(cuda_sources)
endif

cluster_sources =
if USE_CLUSTER
cluster_sources += \
	src/cluster/ClusterManager.cpp \
	src/cluster/ClusterMemoryManagement.cpp \
	src/cluster/messages/Message.cpp \
	src/cluster/messages/MessageDataFetch.cpp \
	src/cluster/messages/MessageDataSend.cpp \
	src/cluster/messages/MessageDfree.cpp \
	src/cluster/messages/MessageDmalloc.cpp \
	src/cluster/messages/MessageId.cpp \
	src/cluster/messages/MessageReleaseAccess.cpp \
	src/cluster/messages/MessageSatisfiability.cpp \
	src/cluster/messages/MessageSysFinish.cpp \
	src/cluster/messages/MessageTaskFinished.cpp \
	src/cluster/messages/MessageTaskNew.cpp \
	src/cluster/messages/MessageType.cpp \
	src/cluster/messenger/mpi/MPIMessenger.cpp \
	src/cluster/offloading/TaskOffloading.cpp \
	src/cluster/polling-services/DataTransferCompletion.cpp \
	src/cluster/polling-services/MessageDelivery.cpp \
	src/cluster/polling-services/MessageHandler.cpp \
	src/executors/workflow/cluster/ExecutionWorkflowCluster.cpp \
	src/memory/directory/cluster/DistributionPolicy.cpp

cluster_cppflags = \
	-I$(srcdir)/src/cluster \
	-I$(srcdir)/src/cluster/messages \
	-I$(srcdir)/src/cluster/messenger \
	-I$(srcdir)/src/cluster/offloading \
	-I$(srcdir)/src/cluster/polling-services \
	-I$(srcdir)/src/executors/workflow/cluster \
	-I$(srcdir)/src/memory/directory \
	-I$(srcdir)/src/memory/directory/cluster \
	-I$(srcdir)/src/hardware/cluster \
	-I$(srcdir)/src/hardware/cluster/memory
else
cluster_cppflags = \
	-I$(srcdir)/src/executors/workflow/cluster/null \
	-I$(srcdir)/src/cluster/null \
	-I$(srcdir)/src/cluster/offloading/null \
	-I$(srcdir)/src/hardware/cluster/null \
	-I$(srcdir)/src/hardware/cluster/memory/null
endif

common_sources += $(cluster_sources)

workflow_sources=
if USE_EXEC_WORKFLOW
workflow_sources += \
	src/executors/workflow/ExecutionStep.cpp \
	src/executors/workflow/ExecutionWorkflow.cpp \
	src/executors/workflow/ExecutionWorkflowHost.cpp
workflow_cppflags = -I$(srcdir)/src/executors/workflow
else
workflow_sources += \
	src/executors/workflow/null/ExecutionWorkflow.cpp
workflow_cppflags = -I$(srcdir)/src/executors/workflow/null
endif

common_sources += ${workflow_sources}

linear_regions_fragmented_dependency_sources = \
	src/dependencies/linear-regions-fragmented/DataAccessRegistration.cpp \
	src/dependencies/linear-regions-fragmented/CommutativeScoreboard.cpp \
	src/dependencies/linear-regions-fragmented/ReductionInfo.cpp \
	src/dependencies/linear-regions-fragmented/Reductions.cpp \
	src/dependencies/linear-regions-fragmented/RegisterDependencies.cpp \
	src/dependencies/linear-regions-fragmented/ReleaseDirective.cpp

multidimensional_adaptor_sources =

if LINEAR_REGION_FRAGMENTED_DEPENDENCIES
AM_CXXFLAGS += -I$(srcdir)/src/dependencies/linear-regions -I$(srcdir)/src/dependencies/linear-regions-fragmented
common_sources += $(linear_regions_fragmented_dependency_sources)
multidimensional_adaptor_sources += MultidimensionalAPIToLinear.cpp
SUPPORTED_MULTIDIMENSIONAL_REGION_API_TYPES += read write readwrite weak_read weak_write weak_readwrite concurrent commutative weak_commutative reduction weak_reduction
endif

introspection_sources = \
	src/instrument/support/introspection/CodeAddressInfoBase.cpp \
	src/instrument/support/introspection/DL/DLCodeAddressInfo.cpp
introspection_cppflags =

if HAVE_ELFUTILS
introspection_sources += \
	src/instrument/support/introspection/ElfUtils/ElfUtilsCodeAddressInfo.cpp
introspection_cppflags += -I$(srcdir)/src/instrument/support/introspection/ElfUtils
else
introspection_sources += \
	src/instrument/support/introspection/Addr2Line/Addr2LineCodeAddressInfo.cpp
introspection_cppflags += -I$(srcdir)/src/instrument/support/introspection/Addr2Line
endif

if HAVE_LIBUNWIND
introspection_cppflags += -I$(srcdir)/src/instrument/support/backtrace/libunwind
else
if HAVE_BACKTRACE
introspection_cppflags += -I$(srcdir)/src/instrument/support/backtrace/backtrace
else
introspection_cppflags += -I$(srcdir)/src/instrument/support/backtrace/unsupported
endif
endif


AM_CXXFLAGS += -I$(srcdir)/src/executors/threads/kernel-level
common_sources += \
	src/executors/threads/kernel-level/CPUThreadingModelData.cpp \
	src/executors/threads/kernel-level/WorkerThreadBase.cpp


nodist_common_sources += \
	$(multidimensional_adaptor_sources)
BUILT_SOURCES += \
	$(multidimensional_adaptor_sources)
CLEANFILES += \
	$(multidimensional_adaptor_sources)


hardware_counters_sources = \
	src/instrument/stats/InstrumentHardwareCounters.cpp

papi_hardware_counters_sources = \
	src/instrument/stats/papi/InstrumentPAPIHardwareCounters.cpp \
	src/instrument/stats/papi/InstrumentPAPIHardwareCountersThreadLocalData.cpp

no_hardware_counters_sources = 

noinstrument_sources = 

instrument_generic_ids_sources = \
	src/instrument/generic_ids/GenericIds.cpp

instrument_graph_sources = \
	$(instrument_generic_ids_sources) \
	src/instrument/graph/ExecutionSteps.cpp \
	src/instrument/graph/GenerateEdges.cpp \
	src/instrument/graph/InstrumentAddTask.cpp \
	src/instrument/graph/InstrumentDependenciesByAccessLinks.cpp \
	src/instrument/graph/InstrumentGraph.cpp \
	src/instrument/graph/InstrumentInitAndShutdown.cpp \
	src/instrument/graph/InstrumentTaskExecution.cpp \
	src/instrument/graph/InstrumentTaskWait.cpp \
	src/instrument/graph/InstrumentUserMutex.cpp \
	src/instrument/graph/PathLength.cpp \
	src/instrument/graph/SortAccessGroups.cpp \
	src/instrument/support/InstrumentThreadInstrumentationContext.cpp \
	src/instrument/support/InstrumentThreadLocalDataSupport.cpp

instrument_profile_sources = \
	src/instrument/profile/InstrumentInitAndShutdown.cpp \
	src/instrument/profile/InstrumentMalloc.cpp \
	src/instrument/profile/InstrumentProfile.cpp \
	src/instrument/support/InstrumentThreadLocalDataSupport.cpp \
	src/instrument/support/sampling/SigProf.cpp

instrument_stats_sources = \
	$(hardware_counters_sources) \
	src/instrument/stats/InstrumentInitAndShutdown.cpp \
	src/instrument/stats/InstrumentStats.cpp \
	src/instrument/support/InstrumentThreadInstrumentationContext.cpp \
	src/instrument/support/InstrumentThreadLocalDataSupport.cpp

instrument_verbose_sources = \
	$(instrument_generic_ids_sources) \
	src/instrument/support/InstrumentThreadInstrumentationContext.cpp \
	src/instrument/support/InstrumentThreadLocalDataSupport.cpp \
	src/instrument/verbose/InstrumentAddTask.cpp \
	src/instrument/verbose/InstrumentBlocking.cpp \
	src/instrument/verbose/InstrumentComputePlaceManagement.cpp \
	src/instrument/verbose/InstrumentDependenciesByAccess.cpp \
	src/instrument/verbose/InstrumentDependenciesByAccessLinks.cpp \
	src/instrument/verbose/InstrumentDependenciesByGroup.cpp \
	src/instrument/verbose/InstrumentInitAndShutdown.cpp \
	src/instrument/verbose/InstrumentLeaderThread.cpp \
	src/instrument/verbose/InstrumentReductions.cpp \
	src/instrument/verbose/InstrumentTaskExecution.cpp \
	src/instrument/verbose/InstrumentTaskStatus.cpp \
	src/instrument/verbose/InstrumentTaskWait.cpp \
	src/instrument/verbose/InstrumentThreadManagement.cpp \
	src/instrument/verbose/InstrumentUserMutex.cpp \
	src/instrument/verbose/InstrumentVerbose.cpp

instrument_extrae_sources = \
	$(instrument_generic_ids_sources) \
	src/instrument/extrae/ExtraeSymbolLiterals.cpp \
	src/instrument/extrae/ExtraeSymbolResolver.cpp \
	src/instrument/extrae/InstrumentExtrae.cpp \
	src/instrument/extrae/InstrumentInitAndShutdown.cpp \
	src/instrument/extrae/InstrumentMalloc.cpp \
	src/instrument/extrae/InstrumentReductions.cpp \
	src/instrument/support/InstrumentThreadInstrumentationContext.cpp \
	src/instrument/support/InstrumentThreadLocalDataSupport.cpp \
	src/instrument/support/sampling/SigProf.cpp

memory_allocator_default_sources = \
	src/memory/allocator/pool/MemoryAllocator.cpp \
	src/memory/allocator/pool/ObjectAllocator.cpp
memory_allocator_default_cppflags = -I$(srcdir)/src/memory/allocator/pool

memory_allocator_debug_sources =
memory_allocator_debug_cppflags = -I$(srcdir)/src/memory/allocator/malloc

vmm_smp_sources = src/memory/vmm/smp/VirtualMemoryManagement.cpp
vmm_smp_cppflags = -I$(srcdir)/src/memory/vmm/smp

vmm_cluster_sources = src/memory/vmm/cluster/VirtualMemoryManagement.cpp
vmm_cluster_cppflags = -I$(srcdir)/src/memory/vmm/cluster

if USE_CLUSTER
memory_default_sources = $(memory_allocator_default_sources) $(vmm_cluster_sources)
memory_default_cppflags = $(memory_allocator_default_cppflags) $(vmm_cluster_cppflags)
memory_debug_sources = $(memory_default_sources)
memory_debug_cppflags = $(memory_default_cppflags)
else
# memory_default_sources = $(memory_allocator_default_sources) $(vmm_smp_sources)
# memory_default_cppflags = $(memory_allocator_default_cppflags) $(vmm_smp_cppflags)
memory_default_sources = $(memory_allocator_debug_sources) $(vmm_smp_sources)
memory_default_cppflags = $(memory_allocator_debug_cppflags) $(vmm_smp_cppflags)
memory_debug_sources = $(memory_allocator_debug_sources) $(vmm_smp_sources)
memory_debug_cppflags = $(memory_allocator_debug_cppflags) $(vmm_smp_cppflags)
endif


# Chronometers
chronometers_sources =
chronometers_CPPFLAGS =
if CHRONO_ARCH
chronometers_sources  += src/support/chronometers/arch/Chrono.cpp
chronometers_CPPFLAGS += -I$(srcdir)/src/support/chronometers/arch
else
chronometers_CPPFLAGS += -I$(srcdir)/src/support/chronometers/std
endif

# Monitoring
monitoring_sources =
monitoring_CPPFLAGS =
if MONITORING_ENABLED
monitoring_sources += \
	src/monitoring/monitoring/CPUMonitor.cpp \
	src/monitoring/monitoring/CPUUsagePredictor.cpp \
	src/monitoring/monitoring/Monitoring.cpp \
	src/monitoring/monitoring/TaskMonitor.cpp \
	src/monitoring/monitoring/TasktypePredictions.cpp \
	src/monitoring/monitoring/WorkloadPredictor.cpp
monitoring_CPPFLAGS += -I$(srcdir)/src/monitoring/monitoring
else
monitoring_CPPFLAGS += -I$(srcdir)/src/monitoring/null
endif

# Monitoring hardware events
hw_monitoring_sources =
hw_monitoring_CPPFLAGS =
hw_monitoring_LDFLAGS =
if MONITORING_HWEVENTS_ENABLED
if HAVE_PQOS
hw_monitoring_LDFLAGS  += $(pqos_LIBS)
hw_monitoring_CPPFLAGS += $(pqos_CPPFLAGS)
hw_monitoring_CPPFLAGS += -I$(srcdir)/src/hardware-counters/pqos
hw_monitoring_sources += \
	src/hardware-counters/pqos/HardwareCounters.cpp \
	src/hardware-counters/pqos/TaskHardwareCountersMonitor.cpp \
	src/hardware-counters/pqos/TasktypeHardwareCountersPredictions.cpp \
	src/hardware-counters/pqos/ThreadHardwareCountersMonitor.cpp
else
hw_monitoring_CPPFLAGS += -I$(srcdir)/src/hardware-counters/null
endif
else
hw_monitoring_CPPFLAGS += -I$(srcdir)/src/hardware-counters/null
endif


# Wisdom manager (monitoring-related)
if MONITORING_ENABLED
monitoring_sources += \
	src/monitoring/support/wisdom/wisdom/WisdomManager.cpp
monitoring_CPPFLAGS += -I$(srcdir)/src/monitoring/support/wisdom/wisdom
endif
if !MONITORING_ENABLED
if MONITORING_HWEVENTS_ENABLED
monitoring_sources += \
	src/monitoring/support/wisdom/wisdom/WisdomManager.cpp
monitoring_CPPFLAGS += -I$(srcdir)/src/monitoring/support/wisdom/wisdom
endif
endif
if !MONITORING_ENABLED
if !MONITORING_HWEVENTS_ENABLED
monitoring_CPPFLAGS += -I$(srcdir)/src/monitoring/support/wisdom/null
endif
endif


# Add monitoring-related sources to common sources
common_sources += $(monitoring_sources) $(chronometers_sources) $(hw_monitoring_sources)


noinst_HEADERS = \
	src/cluster/ClusterManager.hpp \
	src/cluster/ClusterMemoryManagement.hpp \
	src/cluster/messages/Message.hpp \
	src/cluster/messages/MessageDataFetch.hpp \
	src/cluster/messages/MessageDataSend.hpp \
	src/cluster/messages/MessageDfree.hpp \
	src/cluster/messages/MessageDmalloc.hpp \
	src/cluster/messages/MessageId.hpp \
	src/cluster/messages/MessageReleaseAccess.hpp \
	src/cluster/messages/MessageSatisfiability.hpp \
	src/cluster/messages/MessageSysFinish.hpp \
	src/cluster/messages/MessageTaskFinished.hpp \
	src/cluster/messages/MessageTaskNew.hpp \
	src/cluster/messages/MessageType.hpp \
	src/cluster/messenger/DataTransfer.hpp \
	src/cluster/messenger/Messenger.hpp \
	src/cluster/messenger/mpi/MPIDataTransfer.hpp \
	src/cluster/messenger/mpi/MPIMessenger.hpp \
	src/cluster/null/ClusterManager.hpp \
	src/cluster/null/ClusterMemoryManagement.hpp \
	src/cluster/offloading/ClusterTaskContext.hpp \
	src/cluster/offloading/SatisfiabilityInfo.hpp \
	src/cluster/offloading/TaskOffloading.hpp \
	src/cluster/offloading/null/ClusterTaskContext.hpp \
	src/cluster/polling-services/ClusterPollingServices.hpp \
	src/cluster/polling-services/DataTransferCompletion.hpp \
	src/cluster/polling-services/MessageDelivery.hpp \
	src/cluster/polling-services/MessageHandler.hpp \
	src/dependencies/DataAccessBase.hpp \
	src/dependencies/DataAccessType.hpp \
	src/dependencies/MultidimensionalAPITraversal.hpp \
	src/dependencies/linear-regions-fragmented/BottomMapEntry.hpp \
	src/dependencies/linear-regions-fragmented/CommutativeScoreboard.hpp \
	src/dependencies/linear-regions-fragmented/CPUDependencyData.hpp \
	src/dependencies/linear-regions-fragmented/DataAccess.hpp \
	src/dependencies/linear-regions-fragmented/DataAccessLink.hpp \
	src/dependencies/linear-regions-fragmented/DataAccessObjectType.hpp \
	src/dependencies/linear-regions-fragmented/DataAccessRegistration.hpp \
	src/dependencies/linear-regions-fragmented/DataAccessRegistrationImplementation.hpp \
	src/dependencies/linear-regions-fragmented/DependencyDomain.hpp \
	src/dependencies/linear-regions-fragmented/DependencySystem.hpp \
	src/dependencies/linear-regions-fragmented/ReductionInfo.hpp \
	src/dependencies/linear-regions-fragmented/ReductionSpecific.hpp \
	src/dependencies/linear-regions-fragmented/TaskDataAccessHooks.hpp \
	src/dependencies/linear-regions-fragmented/TaskDataAccessLinkingArtifacts.hpp \
	src/dependencies/linear-regions-fragmented/TaskDataAccessLinkingArtifactsImplementation.hpp \
	src/dependencies/linear-regions-fragmented/TaskDataAccesses.hpp \
	src/dependencies/linear-regions-fragmented/TaskDataAccessesImplementation.hpp \
	src/dependencies/linear-regions/DataAccessRegion.hpp \
	src/dependencies/linear-regions/DataAccessRegionIndexer.hpp \
	src/dependencies/linear-regions/Dependencies.hpp \
	src/dependencies/linear-regions/IntrusiveLinearRegionMap.hpp \
	src/dependencies/linear-regions/IntrusiveLinearRegionMapImplementation.hpp \
	src/dependencies/linear-regions/LinearRegionMap.hpp \
	src/dependencies/linear-regions/LinearRegionMapImplementation.hpp \
	src/dependencies/linear-regions/MultidimensionalAPI.hpp \
	src/executors/cuda/CUDAPollingService.hpp \
	src/executors/threads/CPU.hpp \
	src/executors/threads/CPUActivation.hpp \
	src/executors/threads/CPUManager.hpp \
	src/executors/threads/TaskFinalization.hpp \
	src/executors/threads/TaskFinalizationImplementation.hpp \
	src/executors/threads/ThreadManager.hpp \
	src/executors/threads/ThreadManagerPolicy.hpp \
	src/executors/threads/WorkerThread.hpp \
	src/executors/threads/WorkerThreadImplementation.hpp \
	src/executors/threads/kernel-level/CPUThreadingModelData.hpp \
	src/executors/threads/kernel-level/WorkerThreadBase.hpp \
	src/executors/workflow/ExecutionStep.hpp \
	src/executors/workflow/ExecutionWorkflow.hpp \
	src/executors/workflow/ExecutionWorkflowHost.hpp \
	src/executors/workflow/cluster/ExecutionWorkflowCluster.hpp \
	src/executors/workflow/cluster/null/ExecutionWorkflowCluster.hpp \
	src/executors/workflow/null/ExecutionStep.hpp \
	src/executors/workflow/null/ExecutionWorkflow.hpp \
	src/hardware/cluster/ClusterNode.hpp \
	src/hardware/cluster/memory/ClusterMemoryNode.hpp \
	src/hardware/cluster/memory/null/ClusterMemoryNode.hpp \
	src/hardware/cluster/null/ClusterNode.hpp \
	src/hardware/cuda/CUDADevice.hpp \
	src/hardware/cuda/CUDAInfo.hpp \
	src/hardware/cuda/compute/CUDAComputePlace.hpp \
	src/hardware/cuda/compute/stream/CUDAStream.hpp \
	src/hardware/cuda/compute/stream/CUDAStreamPool.hpp \
	src/hardware/cuda/compute/synchronization/CUDAEvent.hpp \
	src/hardware/cuda/compute/synchronization/CUDAEventPool.hpp \
	src/hardware/cuda/memory/CUDAMemoryPlace.hpp \
	src/hardware/HardwareInfo.hpp \
	src/hardware/hwinfo/DeviceInfo.hpp \
	src/hardware/hwinfo/HostInfo.hpp \
	src/hardware/places/CPUPlace.hpp \
	src/hardware/places/ComputePlace.hpp \
	src/hardware/places/MemoryPlace.hpp \
	src/hardware/places/NUMAPlace.hpp \
	src/hardware-counters/SupportedHardwareCounters.hpp \
	src/hardware-counters/null/HardwareCounters.hpp \
	src/hardware-counters/null/TaskHardwareCounters.hpp \
	src/hardware-counters/null/TaskHardwareCountersMonitor.hpp \
	src/hardware-counters/null/TaskHardwareCountersPredictions.hpp \
	src/hardware-counters/null/ThreadHardwareCounters.hpp \
	src/hardware-counters/pqos/HardwareCounters.hpp \
	src/hardware-counters/pqos/TaskHardwareCounters.hpp \
	src/hardware-counters/pqos/TaskHardwareCountersMonitor.hpp \
	src/hardware-counters/pqos/TaskHardwareCountersPredictions.hpp \
	src/hardware-counters/pqos/TasktypeHardwareCountersPredictions.hpp \
	src/hardware-counters/pqos/ThreadHardwareCounters.hpp \
	src/hardware-counters/pqos/ThreadHardwareCountersMonitor.hpp \
	src/instrument/api/InstrumentAddTask.hpp \
	src/instrument/api/InstrumentBlocking.hpp \
	src/instrument/api/InstrumentComputePlaceManagement.hpp \
	src/instrument/api/InstrumentDataAccessTypes.hpp \
	src/instrument/api/InstrumentDependenciesByAccess.hpp \
	src/instrument/api/InstrumentDependenciesByAccessLinks.hpp \
	src/instrument/api/InstrumentDependenciesByGroup.hpp \
	src/instrument/api/InstrumentInitAndShutdown.hpp \
	src/instrument/api/InstrumentLeaderThread.hpp \
	src/instrument/api/InstrumentLogMessage.hpp \
	src/instrument/api/InstrumentReductions.hpp \
	src/instrument/api/InstrumentTaskExecution.hpp \
	src/instrument/api/InstrumentTaskStatus.hpp \
	src/instrument/api/InstrumentTaskWait.hpp \
	src/instrument/api/InstrumentThreadManagement.hpp \
	src/instrument/api/InstrumentTracingPoints.hpp \
	src/instrument/api/InstrumentUserMutex.hpp \
	src/instrument/extrae/ExtraeSymbolLiterals.hpp \
	src/instrument/extrae/ExtraeSymbolResolver.hpp \
	src/instrument/extrae/InstrumentAddTask.hpp \
	src/instrument/extrae/InstrumentBlocking.hpp \
	src/instrument/extrae/InstrumentComputePlaceId.hpp \
	src/instrument/extrae/InstrumentComputePlaceManagement.hpp \
	src/instrument/extrae/InstrumentDataAccessId.hpp \
	src/instrument/extrae/InstrumentDependenciesByAccess.hpp \
	src/instrument/extrae/InstrumentDependenciesByAccessLinks.hpp \
	src/instrument/extrae/InstrumentDependenciesByGroup.hpp \
	src/instrument/extrae/InstrumentExternalThreadId.hpp \
	src/instrument/extrae/InstrumentExternalThreadLocalData.hpp \
	src/instrument/extrae/InstrumentExtrae.hpp \
	src/instrument/extrae/InstrumentInitAndShutdown.hpp \
	src/instrument/extrae/InstrumentLeaderThread.hpp \
	src/instrument/extrae/InstrumentLogMessage.hpp \
	src/instrument/extrae/InstrumentReductions.hpp \
	src/instrument/extrae/InstrumentTaskExecution.hpp \
	src/instrument/extrae/InstrumentTaskId.hpp \
	src/instrument/extrae/InstrumentTaskStatus.hpp \
	src/instrument/extrae/InstrumentTaskWait.hpp \
	src/instrument/extrae/InstrumentThreadId.hpp \
	src/instrument/extrae/InstrumentThreadLocalData.hpp \
	src/instrument/extrae/InstrumentThreadManagement.hpp \
	src/instrument/extrae/InstrumentTracingPointTypes.hpp\
	src/instrument/extrae/InstrumentTracingPoints.hpp \
	src/instrument/extrae/InstrumentUserMutex.hpp \
	src/instrument/extrae/PreloadedExtraeBouncer.hpp \
	src/instrument/generic_ids/GenericIds.hpp \
	src/instrument/generic_ids/InstrumentExternalThreadId.hpp \
	src/instrument/generic_ids/InstrumentThreadId.hpp \
	src/instrument/graph/Color.hpp \
	src/instrument/graph/ExecutionSteps.hpp \
	src/instrument/graph/GenerateEdges.hpp \
	src/instrument/graph/InstrumentAddTask.hpp \
	src/instrument/graph/InstrumentBlocking.hpp \
	src/instrument/graph/InstrumentComputePlaceId.hpp \
	src/instrument/graph/InstrumentComputePlaceManagement.hpp \
	src/instrument/graph/InstrumentDataAccessId.hpp \
	src/instrument/graph/InstrumentDependenciesByAccess.hpp \
	src/instrument/graph/InstrumentDependenciesByAccessLinks.hpp \
	src/instrument/graph/InstrumentDependenciesByGroup.hpp \
	src/instrument/graph/InstrumentExternalThreadId.hpp \
	src/instrument/graph/InstrumentExternalThreadLocalData.hpp \
	src/instrument/graph/InstrumentGraph.hpp \
	src/instrument/graph/InstrumentInitAndShutdown.hpp \
	src/instrument/graph/InstrumentLeaderThread.hpp \
	src/instrument/graph/InstrumentLogMessage.hpp \
	src/instrument/graph/InstrumentReductions.hpp \
	src/instrument/graph/InstrumentTaskExecution.hpp \
	src/instrument/graph/InstrumentTaskId.hpp \
	src/instrument/graph/InstrumentTaskStatus.hpp \
	src/instrument/graph/InstrumentTaskWait.hpp \
	src/instrument/graph/InstrumentThreadId.hpp \
	src/instrument/graph/InstrumentThreadLocalData.hpp \
	src/instrument/graph/InstrumentThreadManagement.hpp \
	src/instrument/graph/InstrumentTracingPointTypes.hpp \
	src/instrument/graph/InstrumentTracingPoints.hpp \
	src/instrument/graph/InstrumentUserMutex.hpp \
	src/instrument/graph/PathLength.hpp \
	src/instrument/graph/SortAccessGroups.hpp \
	src/instrument/null/InstrumentAddTask.hpp \
	src/instrument/null/InstrumentBlocking.hpp \
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
	src/instrument/null/InstrumentReductions.hpp \
	src/instrument/null/InstrumentTaskExecution.hpp \
	src/instrument/null/InstrumentTaskId.hpp \
	src/instrument/null/InstrumentTaskStatus.hpp \
	src/instrument/null/InstrumentTaskWait.hpp \
	src/instrument/null/InstrumentThreadId.hpp \
	src/instrument/null/InstrumentThreadInstrumentationContext.hpp \
	src/instrument/null/InstrumentThreadLocalData.hpp \
	src/instrument/null/InstrumentThreadManagement.hpp \
	src/instrument/null/InstrumentTracingPointTypes.hpp \
	src/instrument/null/InstrumentTracingPoints.hpp \
	src/instrument/null/InstrumentUserMutex.hpp \
	src/instrument/profile/Address.hpp \
	src/instrument/profile/InstrumentAddTask.hpp \
	src/instrument/profile/InstrumentBlocking.hpp \
	src/instrument/profile/InstrumentComputePlaceId.hpp \
	src/instrument/profile/InstrumentComputePlaceManagement.hpp \
	src/instrument/profile/InstrumentDataAccessId.hpp \
	src/instrument/profile/InstrumentDependenciesByAccess.hpp \
	src/instrument/profile/InstrumentDependenciesByAccessLinks.hpp \
	src/instrument/profile/InstrumentDependenciesByGroup.hpp \
	src/instrument/profile/InstrumentExternalThreadId.hpp \
	src/instrument/profile/InstrumentExternalThreadLocalData.hpp \
	src/instrument/profile/InstrumentInitAndShutdown.hpp \
	src/instrument/profile/InstrumentInstrumentationContext.hpp \
	src/instrument/profile/InstrumentLeaderThread.hpp \
	src/instrument/profile/InstrumentLogMessage.hpp \
	src/instrument/profile/InstrumentReductions.hpp \
	src/instrument/profile/InstrumentProfile.hpp \
	src/instrument/profile/InstrumentTaskExecution.hpp \
	src/instrument/profile/InstrumentTaskId.hpp \
	src/instrument/profile/InstrumentTaskStatus.hpp \
	src/instrument/profile/InstrumentTaskWait.hpp \
	src/instrument/profile/InstrumentThreadId.hpp \
	src/instrument/profile/InstrumentThreadInstrumentationContext.hpp \
	src/instrument/profile/InstrumentThreadLocalData.hpp \
	src/instrument/profile/InstrumentThreadManagement.hpp \
	src/instrument/profile/InstrumentTracingPointTypes.hpp \
	src/instrument/profile/InstrumentTracingPoints.hpp \
	src/instrument/profile/InstrumentUserMutex.hpp \
	src/instrument/stats/InstrumentAddTask.hpp \
	src/instrument/stats/InstrumentBlocking.hpp \
	src/instrument/stats/InstrumentComputePlaceId.hpp \
	src/instrument/stats/InstrumentComputePlaceManagement.hpp \
	src/instrument/stats/InstrumentDataAccessId.hpp \
	src/instrument/stats/InstrumentDependenciesByAccess.hpp \
	src/instrument/stats/InstrumentDependenciesByAccessLinks.hpp \
	src/instrument/stats/InstrumentDependenciesByGroup.hpp \
	src/instrument/stats/InstrumentExternalThreadId.hpp \
	src/instrument/stats/InstrumentExternalThreadLocalData.hpp \
	src/instrument/stats/InstrumentHardwareCounters.hpp \
	src/instrument/stats/InstrumentHardwareCountersThreadLocalData.hpp \
	src/instrument/stats/InstrumentInitAndShutdown.hpp \
	src/instrument/stats/InstrumentLeaderThread.hpp \
	src/instrument/stats/InstrumentLogMessage.hpp \
	src/instrument/stats/InstrumentReductions.hpp \
	src/instrument/stats/InstrumentStats.hpp \
	src/instrument/stats/InstrumentTaskExecution.hpp \
	src/instrument/stats/InstrumentTaskId.hpp \
	src/instrument/stats/InstrumentTaskStatus.hpp \
	src/instrument/stats/InstrumentTaskWait.hpp \
	src/instrument/stats/InstrumentThreadId.hpp \
	src/instrument/stats/InstrumentThreadLocalData.hpp \
	src/instrument/stats/InstrumentThreadManagement.hpp \
	src/instrument/stats/InstrumentTracingPointTypes.hpp \
	src/instrument/stats/InstrumentTracingPoints.hpp \
	src/instrument/stats/InstrumentUserMutex.hpp \
	src/instrument/stats/Timer.hpp \
	src/instrument/stats/null/InstrumentNullHardwareCounters.hpp \
	src/instrument/stats/null/InstrumentNullHardwareCountersThreadLocalData.hpp \
	src/instrument/stats/papi/InstrumentPAPIHardwareCounters.hpp \
	src/instrument/stats/papi/InstrumentPAPIHardwareCountersThreadLocalData.hpp \
	src/instrument/support/InstrumentHardwarePlaceManagement.hpp \
	src/instrument/support/InstrumentInstrumentationContext.hpp \
	src/instrument/support/InstrumentStandardExternalThreadLocalData.hpp \
	src/instrument/support/InstrumentThreadInstrumentationContext.hpp \
	src/instrument/support/InstrumentThreadLocalDataSupport.hpp \
	src/instrument/support/backtrace/backtrace/BacktraceWalker.hpp \
	src/instrument/support/backtrace/libunwind/BacktraceWalker.hpp \
	src/instrument/support/backtrace/unsupported/BacktraceWalker.hpp \
	src/instrument/support/introspection/Addr2Line/Addr2LineCodeAddressInfo.hpp \
	src/instrument/support/introspection/Addr2Line/CodeAddressInfo.hpp \
	src/instrument/support/introspection/CodeAddressInfoBase.hpp \
	src/instrument/support/introspection/DL/CodeAddressInfo.hpp \
	src/instrument/support/introspection/DL/DLCodeAddressInfo.hpp \
	src/instrument/support/introspection/ElfUtils/CodeAddressInfo.hpp \
	src/instrument/support/introspection/ElfUtils/ElfUtilsCodeAddressInfo.hpp \
	src/instrument/support/sampling/SigProf.hpp \
	src/instrument/support/sampling/ThreadLocalData.hpp \
	src/instrument/verbose/InstrumentAddTask.hpp \
	src/instrument/verbose/InstrumentBlocking.hpp \
	src/instrument/verbose/InstrumentComputePlaceId.hpp \
	src/instrument/verbose/InstrumentComputePlaceManagement.hpp \
	src/instrument/verbose/InstrumentDataAccessId.hpp \
	src/instrument/verbose/InstrumentDependenciesByAccess.hpp \
	src/instrument/verbose/InstrumentDependenciesByAccessLinks.hpp \
	src/instrument/verbose/InstrumentDependenciesByGroup.hpp \
	src/instrument/verbose/InstrumentExternalThreadId.hpp \
	src/instrument/verbose/InstrumentExternalThreadLocalData.hpp \
	src/instrument/verbose/InstrumentInitAndShutdown.hpp \
	src/instrument/verbose/InstrumentLeaderThread.hpp \
	src/instrument/verbose/InstrumentLogMessage.hpp \
	src/instrument/verbose/InstrumentReductions.hpp \
	src/instrument/verbose/InstrumentTaskExecution.hpp \
	src/instrument/verbose/InstrumentTaskId.hpp \
	src/instrument/verbose/InstrumentTaskStatus.hpp \
	src/instrument/verbose/InstrumentTaskWait.hpp \
	src/instrument/verbose/InstrumentThreadId.hpp \
	src/instrument/verbose/InstrumentThreadLocalData.hpp \
	src/instrument/verbose/InstrumentThreadManagement.hpp \
	src/instrument/verbose/InstrumentTracingPointTypes.hpp \
	src/instrument/verbose/InstrumentTracingPoints.hpp \
	src/instrument/verbose/InstrumentUserMutex.hpp \
	src/instrument/verbose/InstrumentVerbose.hpp \
	src/lowlevel/BacktraceRecording.hpp \
	src/lowlevel/ConditionVariable.hpp \
	src/lowlevel/EnvironmentVariable.hpp \
	src/lowlevel/FatalErrorHandler.hpp \
	src/lowlevel/PaddedSpinLock.hpp \
	src/lowlevel/PaddedTicketSpinLock.hpp \
	src/lowlevel/RWSpinLock.hpp \
	src/lowlevel/RWTicketSpinLock.hpp \
	src/lowlevel/SpinLock.hpp \
	src/lowlevel/SpinLockBacktracingDebug.hpp \
	src/lowlevel/SpinLockDeadlockDebug.hpp \
	src/lowlevel/SpinLockNoDebug.hpp \
	src/lowlevel/SpinLockOwnerDebug.hpp \
	src/lowlevel/SymbolResolver.hpp \
	src/lowlevel/TicketSpinLock.hpp \
	src/lowlevel/TokenizedEnvironmentVariable.hpp \
	src/lowlevel/apple/SpinLockImplementation.hpp \
	src/lowlevel/cuda/CUDAErrorHandler.hpp \
	src/lowlevel/cxx/SpinLockImplementation.hpp \
	src/lowlevel/mpi/MPIErrorHandler.hpp \
	src/lowlevel/posix/SpinLockImplementation.hpp \
	src/lowlevel/threads/ExternalThread.hpp \
	src/lowlevel/threads/ExternalThreadGroup.hpp \
	src/lowlevel/threads/HelperThread.hpp \
	src/lowlevel/threads/KernelLevelThread.hpp \
	src/lowlevel/threads/posix/KernelLevelThread.hpp \
	src/memory/AddressSpace.hpp \
	src/memory/allocator/malloc/MemoryAllocator.hpp \
	src/memory/allocator/malloc/ObjectAllocator.hpp \
	src/memory/allocator/pool/CPUObjectCache.hpp \
	src/memory/allocator/pool/ExternalMemoryPool.hpp \
	src/memory/allocator/pool/MemoryAllocator.hpp \
	src/memory/allocator/pool/MemoryPool.hpp \
	src/memory/allocator/pool/MemoryPoolGlobal.hpp \
	src/memory/allocator/pool/NUMAObjectCache.hpp \
	src/memory/allocator/pool/ObjectAllocator.hpp \
	src/memory/allocator/pool/ObjectCache.hpp \
	src/memory/directory/Directory.hpp \
	src/memory/directory/HomeMapEntry.hpp \
	src/memory/directory/HomeNodeMap.hpp \
	src/memory/directory/cluster/DistributionPolicy.hpp \
	src/memory/vmm/VirtualMemoryAllocation.hpp \
	src/memory/vmm/VirtualMemoryArea.hpp \
	src/memory/vmm/cluster/VirtualMemoryManagement.hpp \
	src/memory/vmm/smp/VirtualMemoryManagement.hpp \
	src/monitoring/monitoring/CPUMonitor.hpp \
	src/monitoring/monitoring/CPUStatistics.hpp \
	src/monitoring/monitoring/CPUUsagePredictor.hpp \
	src/monitoring/monitoring/Monitoring.hpp \
	src/monitoring/monitoring/TaskMonitor.hpp \
	src/monitoring/monitoring/TaskPredictions.hpp \
	src/monitoring/monitoring/TaskStatistics.hpp \
	src/monitoring/monitoring/TasktypePredictions.hpp \
	src/monitoring/monitoring/WorkloadPredictor.hpp \
	src/monitoring/monitoring/WorkloadStatistics.hpp \
	src/monitoring/null/Monitoring.hpp \
	src/monitoring/null/TaskMonitor.hpp \
	src/monitoring/null/TaskPredictions.hpp \
	src/monitoring/null/TaskStatistics.hpp \
	src/monitoring/support/wisdom/null/WisdomManager.hpp \
	src/monitoring/support/wisdom/wisdom/WisdomManager.hpp \
	src/scheduling/Scheduler.hpp \
	src/scheduling/SchedulerGenerator.hpp \
	src/scheduling/SchedulerInterface.hpp \
	src/scheduling/TaskloopSchedulingPolicy.hpp \
	src/scheduling/schedulers/DefaultScheduler.hpp \
	src/scheduling/schedulers/DeviceHierarchicalScheduler.hpp \
	src/scheduling/schedulers/FIFOImmediateSuccessorWithPollingScheduler.hpp \
	src/scheduling/schedulers/FIFOScheduler.hpp \
	src/scheduling/schedulers/HostHierarchicalScheduler.hpp \
	src/scheduling/schedulers/ImmediateSuccessorScheduler.hpp \
	src/scheduling/schedulers/ImmediateSuccessorWithPollingScheduler.hpp \
	src/scheduling/schedulers/NUMAHierarchicalScheduler.hpp \
	src/scheduling/schedulers/NaiveScheduler.hpp \
	src/scheduling/schedulers/NoSleepPriorityScheduler.hpp \
	src/scheduling/schedulers/PriorityScheduler.hpp \
	src/scheduling/schedulers/PriorityScheduler1.hpp \
	src/scheduling/schedulers/cluster/ClusterLocalityScheduler.hpp \
	src/scheduling/schedulers/cluster/ClusterRandomScheduler.hpp \
	src/scheduling/schedulers/cuda/CUDANaiveScheduler.hpp \
	src/support/ConcurrentUnorderedList.hpp \
	src/support/ConstPropagator.hpp \
	src/support/GenericFactory.hpp \
	src/support/GlobalLock.hpp \
	src/support/InlineDoublyLinkedList.hpp \
	src/support/InstrumentedThread.hpp \
	src/support/Objectified.hpp \
	src/support/StringComposer.hpp \
	src/support/StringLiteral.hpp \
	src/support/chronometers/arch/Chrono.hpp \
	src/support/chronometers/std/Chrono.hpp \
	src/system/APICheck.hpp \
	src/system/If0Task.hpp \
	src/system/LeaderThread.hpp \
	src/system/PollingAPI.hpp \
	src/system/RuntimeInfo.hpp \
	src/system/RuntimeInfoEssentials.hpp \
	src/system/ompss/SpawnFunction.hpp \
	src/system/ompss/TaskBlocking.hpp \
	src/system/ompss/UserMutex.hpp \
	src/tasks/Task.hpp \
	src/tasks/TaskDeviceData.hpp \
	src/tasks/TaskDebuggingInterface.hpp \
	src/tasks/TaskImplementation.hpp \
	src/tasks/Taskloop.hpp \
	src/tasks/TaskloopGenerator.hpp \
	src/tasks/TaskloopInfo.hpp \
	src/tasks/TaskloopLogic.hpp \
	src/version/VersionInfo.hpp \
	tests/Atomic.hpp \
	tests/Functors.hpp \
	tests/TestAnyProtocolProducer.hpp \
	tests/Timer.hpp


EXTRA_DIST += \
	tests/select-version.sh \
	tests/tap-driver.pl \
	tests/tap-driver.sh

common_libnanos6_cppflags = $(BOOST_CPPFLAGS) -DBOOST_ENABLE_ASSERT_DEBUG_HANDLER $(PTHREAD_CFLAGS) $(hwloc_CFLAGS) $(libnuma_CPPFLAGS) $(CUDA_CFLAGS) $(MPI_CXXFLAGS) $(memkind_CPPFLAGS) $(cluster_cppflags) $(workflow_cppflags) $(chronometers_CPPFLAGS) $(monitoring_CPPFLAGS) $(hw_monitoring_CPPFLAGS)
common_libnanos6_ldflags = $(AM_LDFLAGS) $(BOOST_LDFLAGS) -version-info $(lib_current):$(lib_revision):$(lib_age) $(PTHREAD_CFLAGS) $(PTHREAD_LIBS) $(LDFLAGS_NOUNDEFINED) $(hwloc_LIBS) $(libnuma_LIBS) $(DLOPEN_LIBS) $(CUDA_LIBS) $(MPI_CXXLDFLAGS) $(memkind_LIBS) $(hw_monitoring_LDFLAGS)


# This is needed for debuggable spinlocks
common_sources += $(introspection_sources)
AM_CXXFLAGS += $(introspection_cppflags) $(elfutils_CFLAGS) 
common_libnanos6_ldflags  += $(BACKTRACE_LIBS) $(elfutils_LIBS)



enabled_variants =
disabled_variants =


libnanos6_optimized_la_CPPFLAGS = -DNDEBUG $(common_libnanos6_cppflags) $(memory_default_cppflags) -I$(srcdir)/src/instrument/null
libnanos6_optimized_la_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
libnanos6_optimized_la_LDFLAGS = $(common_libnanos6_ldflags)
libnanos6_optimized_la_SOURCES =
nodist_libnanos6_optimized_la_SOURCES =

if BUILD_OPTIMIZED_VARIANT
enabled_variants += optimized
libnanos6_optimized_la_SOURCES += $(common_sources) $(noinstrument_sources) $(universal_debug_sources) $(memory_default_sources)
nodist_libnanos6_optimized_la_SOURCES += $(nodist_common_sources) $(nodist_universal_debug_sources)
else
disabled_variants += optimized
libnanos6_optimized_la_SOURCES += loader/disabled_variant.c
endif


libnanos6_debug_la_CPPFLAGS = $(common_libnanos6_cppflags) $(memory_debug_cppflags) -I$(srcdir)/src/instrument/null
libnanos6_debug_la_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
libnanos6_debug_la_LDFLAGS = $(common_libnanos6_ldflags)
libnanos6_debug_la_SOURCES =
nodist_libnanos6_debug_la_SOURCES =

if BUILD_DEBUG_VARIANTS
enabled_variants += debug
libnanos6_debug_la_SOURCES += $(common_sources) $(noinstrument_sources) $(universal_debug_sources) $(memory_debug_sources) src/instrument/null/InstrumentMalloc.cpp
nodist_libnanos6_debug_la_SOURCES += $(nodist_common_sources) $(nodist_universal_debug_sources)
else
disabled_variants += debug
libnanos6_debug_la_SOURCES += loader/disabled_variant.c
endif


libnanos6_graph_la_CPPFLAGS = -DNDEBUG $(common_libnanos6_cppflags) $(memory_default_cppflags) -I$(srcdir)/src/instrument/graph -I$(srcdir)/src/instrument/support
libnanos6_graph_la_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
libnanos6_graph_la_LDFLAGS = $(common_libnanos6_ldflags)
libnanos6_graph_la_SOURCES =
nodist_libnanos6_graph_la_SOURCES =

if BUILD_GRAPH_INSTRUMENTATION_VARIANT
enabled_variants += graph
libnanos6_graph_la_SOURCES += $(common_sources) $(instrument_graph_sources) $(universal_debug_sources) $(memory_default_sources)
nodist_libnanos6_graph_la_SOURCES += $(nodist_common_sources) $(nodist_universal_debug_sources)
else
disabled_variants += graph
libnanos6_graph_la_SOURCES += loader/disabled_variant.c
endif


libnanos6_graph_debug_la_CPPFLAGS = $(common_libnanos6_cppflags) $(memory_debug_cppflags) -I$(srcdir)/src/instrument/graph -I$(srcdir)/src/instrument/support
libnanos6_graph_debug_la_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
libnanos6_graph_debug_la_LDFLAGS = $(common_libnanos6_ldflags)
libnanos6_graph_debug_la_SOURCES =
nodist_libnanos6_graph_debug_la_SOURCES =

if BUILD_GRAPH_INSTRUMENTATION_VARIANT
if BUILD_DEBUG_VARIANTS
enabled_variants += graph-debug
libnanos6_graph_debug_la_SOURCES += $(common_sources) $(instrument_graph_sources) $(universal_debug_sources) $(memory_debug_sources)
nodist_libnanos6_graph_debug_la_SOURCES += $(nodist_common_sources) $(nodist_universal_debug_sources)
else
disabled_variants += graph-debug
libnanos6_graph_debug_la_SOURCES += loader/disabled_variant.c
endif
else
disabled_variants += graph-debug
libnanos6_graph_debug_la_SOURCES += loader/disabled_variant.c
endif


libnanos6_profile_la_CPPFLAGS = -DNDEBUG $(common_libnanos6_cppflags) $(memory_default_cppflags) -I$(srcdir)/src/instrument/profile $(elfutils_CFLAGS) $(introspection_cppflags)
libnanos6_profile_la_CXXFLAGS = $(PROFILE_CXXFLAGS) $(AM_CXXFLAGS) -g2
libnanos6_profile_la_LDFLAGS = $(common_libnanos6_ldflags) $(BACKTRACE_LIBS) $(CLOCK_LIBS) $(elfutils_LIBS)
libnanos6_profile_la_SOURCES =
nodist_libnanos6_profile_la_SOURCES =

if BUILD_PROFILE_INSTRUMENTATION_VARIANT
enabled_variants += profile
libnanos6_profile_la_SOURCES += $(common_sources) $(instrument_profile_sources) $(universal_debug_sources) $(memory_default_sources)
nodist_libnanos6_profile_la_SOURCES += $(nodist_common_sources) $(nodist_universal_debug_sources)
else
disabled_variants += profile
libnanos6_profile_la_SOURCES += loader/disabled_variant.c
endif


libnanos6_stats_la_CPPFLAGS = -DNDEBUG $(common_libnanos6_cppflags) $(memory_default_cppflags) -I$(srcdir)/src/instrument/stats -I$(srcdir)/src/instrument/support
libnanos6_stats_la_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS) -g2
libnanos6_stats_la_LDFLAGS = $(common_libnanos6_ldflags) $(CLOCK_LIBS)
libnanos6_stats_la_SOURCES =
nodist_libnanos6_stats_la_SOURCES =
if BUILD_STATS_INSTRUMENTATION_VARIANT
enabled_variants += stats
libnanos6_stats_la_SOURCES += $(common_sources) $(instrument_stats_sources) $(no_hardware_counters_sources) $(universal_debug_sources) $(memory_default_sources)
nodist_libnanos6_stats_la_SOURCES += $(nodist_common_sources) $(nodist_universal_debug_sources)
else
disabled_variants += stats
libnanos6_stats_la_SOURCES += loader/disabled_variant.c
endif


libnanos6_stats_papi_la_CPPFLAGS = -DNDEBUG $(common_libnanos6_cppflags) $(memory_default_cppflags) -I$(srcdir)/src/instrument/stats -I$(srcdir)/src/instrument/support $(papi_CPPFLAGS) -DHAVE_PAPI
libnanos6_stats_papi_la_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS) -g2
libnanos6_stats_papi_la_LDFLAGS = $(common_libnanos6_ldflags) $(papi_LIBS) $(CLOCK_LIBS)
libnanos6_stats_papi_la_SOURCES =
nodist_libnanos6_stats_papi_la_SOURCES =

if BUILD_STATS_INSTRUMENTATION_VARIANT
if HAVE_PAPI
enabled_variants += stats-papi
libnanos6_stats_papi_la_SOURCES += $(common_sources) $(instrument_stats_sources) $(papi_sources) $(papi_hardware_counters_sources) $(universal_debug_sources) $(memory_default_sources)
nodist_libnanos6_stats_papi_la_SOURCES += $(nodist_common_sources) $(nodist_universal_debug_sources)
else
disabled_variants += stats-papi
libnanos6_stats_papi_la_SOURCES += loader/disabled_variant.c
endif
else
disabled_variants += stats-papi
libnanos6_stats_papi_la_SOURCES += loader/disabled_variant.c
endif


libnanos6_verbose_la_CPPFLAGS = -DNDEBUG $(common_libnanos6_cppflags) $(memory_default_cppflags) -I$(srcdir)/src/instrument/verbose -I$(srcdir)/src/instrument/support
libnanos6_verbose_la_CXXFLAGS = $(OPT_CXXFLAGS) $(AM_CXXFLAGS)
libnanos6_verbose_la_LDFLAGS = $(common_libnanos6_ldflags) $(CLOCK_LIBS) $(ANDROID_LOG_LIBS)
libnanos6_verbose_la_SOURCES =
nodist_libnanos6_verbose_la_SOURCES =

if BUILD_VERBOSE_INSTRUMENTATION_VARIANT
enabled_variants += verbose
libnanos6_verbose_la_SOURCES += $(common_sources) $(instrument_verbose_sources) $(universal_debug_sources) $(memory_default_sources)
nodist_libnanos6_verbose_la_SOURCES += $(nodist_common_sources) $(nodist_universal_debug_sources)
else
disabled_variants += verbose
nodist_libnanos6_verbose_la_SOURCES += loader/disabled_variant.c
endif


libnanos6_verbose_debug_la_CPPFLAGS = $(common_libnanos6_cppflags) $(memory_debug_cppflags) -I$(srcdir)/src/instrument/verbose -I$(srcdir)/src/instrument/support
libnanos6_verbose_debug_la_CXXFLAGS = $(DEBUG_CXXFLAGS) $(AM_CXXFLAGS)
libnanos6_verbose_debug_la_LDFLAGS = $(common_libnanos6_ldflags) $(CLOCK_LIBS) $(ANDROID_LOG_LIBS)
libnanos6_verbose_debug_la_SOURCES =
nodist_libnanos6_verbose_debug_la_SOURCES =

if BUILD_VERBOSE_INSTRUMENTATION_VARIANT
if BUILD_DEBUG_VARIANTS
enabled_variants += verbose-debug
libnanos6_verbose_debug_la_SOURCES += $(common_sources) $(instrument_verbose_sources) $(universal_debug_sources) $(memory_debug_sources)
nodist_libnanos6_verbose_debug_la_SOURCES += $(nodist_common_sources) $(nodist_universal_debug_sources)
else
disabled_variants += verbose-debug
libnanos6_verbose_debug_la_SOURCES += loader/disabled_variant.c
endif
else
disabled_variants += verbose-debug
libnanos6_verbose_debug_la_SOURCES += loader/disabled_variant.c
endif


libnanos6_extrae_la_CPPFLAGS = -DNDEBUG $(common_libnanos6_cppflags) $(memory_default_cppflags) -I$(srcdir)/src/instrument/extrae -I$(srcdir)/src/instrument/support @extrae_CPPFLAGS@ $(elfutils_CFLAGS) $(introspection_cppflags)
libnanos6_extrae_la_CXXFLAGS = $(PROFILE_CXXFLAGS) $(AM_CXXFLAGS) -g2
libnanos6_extrae_la_LDFLAGS = $(common_libnanos6_ldflags) @extrae_LDFLAGS@ $(BACKTRACE_LIBS) $(CLOCK_LIBS) $(elfutils_LIBS)
libnanos6_extrae_la_SOURCES =
nodist_libnanos6_extrae_la_SOURCES =

if HAVE_EXTRAE
if BUILD_EXTRAE_INSTRUMENTATION_VARIANT
enabled_variants += extrae
libnanos6_extrae_la_SOURCES += $(common_sources) $(instrument_extrae_sources) $(universal_debug_sources) $(memory_default_sources)
nodist_libnanos6_extrae_la_SOURCES += $(nodist_common_sources) $(nodist_universal_debug_sources)
else
disabled_variants += extrae
libnanos6_extrae_la_SOURCES += loader/disabled_variant.c
endif
else
disabled_variants += extrae
libnanos6_extrae_la_SOURCES += loader/disabled_variant.c
endif


all-local: $(lib_OBJECTS)


install-exec-local: $(lib_OBJECTS)
	@$(NORMAL_INSTALL)
	@list='$(lib_OBJECTS)'; test -n "$(libdir)" || list=; \
	list2=; for p in $$list; do \
		if test -f $$p; then \
		list2="$$list2 $$p"; \
		else :; fi; \
	done; \
	test -z "$$list2" || { \
		echo " $(MKDIR_P) '$(DESTDIR)$(libdir)'"; \
		$(MKDIR_P) "$(DESTDIR)$(libdir)" || exit 1; \
		echo " $(INSTALL) $$list2 '$(DESTDIR)$(libdir)'"; \
		$(INSTALL) $$list2 "$(DESTDIR)$(libdir)"; \
	}
	@echo Removing deprecated headers
	for f in nanos6_rt_interface.h nanos6_debug_interface.h dependencies.h ; do \
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
	@echo Removing deprecated helper library
	for l in libnanos6-main-wrapper.a ; do \
		if test -f "$(DESTDIR)$(libdir)/$$l" ; then \
			rm -f "$(DESTDIR)$(libdir)/$$l" ; \
		fi ; \
	done


uninstall-hook:
	for l in $(lib_OBJECTS) ; do \
		if test -f "$(DESTDIR)$(libdir)/$$l" ; then \
			$(LIBTOOL) $(AM_LIBTOOLFLAGS) $(LIBTOOLFLAGS) --mode=uninstall rm -f "$(DESTDIR)$(libdir)/$$l" ; \
		fi ; \
	done
	@echo Removing deprecated headers
	for f in nanos6_rt_interface.h nanos6_debug_interface.h dependencies.h ; do \
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
	@echo Removing deprecated helper library
	for l in libnanos6-main-wrapper.a ; do \
		if test -f "$(DESTDIR)$(libdir)/$$l" ; then \
			rm -f "$(DESTDIR)$(libdir)/$$l" ; \
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
	echo 'char const * nanos6_copyright = "2015-2018 Barcelona Supercomputing Center (BSC)";' >> src/version/CodeVersionInfo.cpp ; \
	echo >> src/version/CodeVersionInfo.cpp ; \
	\
	echo 'char const * nanos6_license = "GPL3";' >> src/version/CodeVersionInfo.cpp ; \
	echo >> src/version/CodeVersionInfo.cpp ; \
	\
	echo -n 'char const * nanos6_full_license = "' >> src/version/CodeVersionInfo.cpp ; \
	cat $(top_srcdir)/COPYING | sed 's@\\@\\\\@g;s@"@\\"@g' | awk 'BEGIN {ORS="\\n";} {print $$0;}' >> src/version/CodeVersionInfo.cpp ; \
	echo '";' >> src/version/CodeVersionInfo.cpp ; \
	echo >> src/version/CodeVersionInfo.cpp
else
src/version/CodeVersionInfo.cpp:
	@echo Error: this Nanos6 distribution does not come neither from a git repository nor a "make dist" tarball.
	@exit 1
endif


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
		$$(find $(top_srcdir)/m4 -name '*.m4') \
		$(nodist_nanos6include_HEADERS) \
		$(top_srcdir)/loader/scripts/*.sh \
	; do \
		if ! grep Copyright $$f > /dev/null ; then \
			echo Error: $$f is missing the copyright notice 1>&2 ; \
			fail=true; \
		fi ; \
	done ; \
	test -z $$fail


show-test-env:
	@echo env LD_LIBRARY_PATH=\"$$(readlink -f $(top_builddir)/.libs):"$$"{LD_LIBRARY_PATH}\"

show-enabled-variants:
	@echo $(enabled_variants)

show-disabled-variants:
	@echo $(disabled_variants)


paraverconfigdir = ${docdir}
nobase_dist_paraverconfig_DATA = \
	paraver-cfg/nanos6/00_overview.cfg \
	paraver-cfg/nanos6/01_overview_histograms.cfg \
	paraver-cfg/nanos6/02_overview_hardware_counters.cfg \
	paraver-cfg/nanos6/03_overview_hardware_counter_histograms.cfg \
	paraver-cfg/nanos6/advanced/2d_general.cfg \
	paraver-cfg/nanos6/advanced/2dh_effective_paralelism.cfg \
	paraver-cfg/nanos6/advanced/3dh_cycles_per_us_by_runtime_state.cfg \
	paraver-cfg/nanos6/advanced/3dh_cycles_per_us_by_task.cfg \
	paraver-cfg/nanos6/advanced/3dh_instructions_vs_cycles.cfg \
	paraver-cfg/nanos6/advanced/3dh_runtime_state_duration.cfg \
	paraver-cfg/nanos6/advanced/3dh_task_duration.cfg \
	paraver-cfg/nanos6/advanced/3dh_task_frequency.cfg \
	paraver-cfg/nanos6/advanced/3dh_task_instructions.cfg \
	paraver-cfg/nanos6/advanced/3dh_task_ipc.cfg \
	paraver-cfg/nanos6/advanced/3dh_task_l2d_miss_ratio.cfg \
	paraver-cfg/nanos6/advanced/cpu.cfg \
	paraver-cfg/nanos6/advanced/effective_paralelism.cfg \
	paraver-cfg/nanos6/advanced/instantiating_function_name_and_code_line.cfg \
	paraver-cfg/nanos6/advanced/nesting_level.cfg \
	paraver-cfg/nanos6/advanced/numa_node.cfg \
	paraver-cfg/nanos6/advanced/reference_clock.cfg \
	paraver-cfg/nanos6/advanced/running_function_name_and_code_line.cfg \
	paraver-cfg/nanos6/advanced/runtime_state.cfg \
	paraver-cfg/nanos6/advanced/thread.cfg \
	paraver-cfg/nanos6/advanced/thread_creation.cfg \
	paraver-cfg/nanos6/advanced/thread_creation_and_runtime_state.cfg \
	paraver-cfg/nanos6/advanced/thread_creation_latency.cfg \
	paraver-cfg/nanos6/graph_and_scheduling/creating_submitting_task.cfg \
	paraver-cfg/nanos6/graph_and_scheduling/nb_ready_tasks.cfg \
	paraver-cfg/nanos6/graph_and_scheduling/nb_tasks_in_graph.cfg \
	paraver-cfg/nanos6/graph_and_scheduling/task_being_created.cfg \
	paraver-cfg/nanos6/tasks/3dh_duration_task.cfg \
	paraver-cfg/nanos6/tasks/3dh_instr_task.cfg \
	paraver-cfg/nanos6/tasks/3dh_ipc_task.cfg \
	paraver-cfg/nanos6/tasks/task_and_deps.cfg \
	paraver-cfg/nanos6/tasks/task_number.cfg \
	paraver-cfg/nanos6/tasks/task_priority.cfg


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
