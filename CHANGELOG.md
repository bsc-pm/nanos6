# Nanos6 Release Notes
All notable changes to this project will be documented in this file.

## Version 2.2, Mon Jun 17, 2019
The 2.2 release corresponds to the OmpSs-2 2019.06 release. It mainly introduces the new support for OmpSs-2@Cluster. It also
includes some improvements and optimizations for array task reductions and general bugfixes.

### General
- Support for OmpSs-2@Cluster
- Bugfixes and performance improvements

### Dependency System
- Bugfixes and optimization for array reductions
- Delete obsolete task data dependency implementations

### Scheduling
- Delete obsolete schedulers

## Version 2.1, Fri Nov 9, 2018
The 2.1 release corresponds to the OmpSs-2 2018.11 release. It provides full support for the [TAMPI](https://github.com/bsc-pm/tampi)
library. It also includes general bugfixes and performance improvements.

### General
- Full support for TAMPI
- Bugfixes and performance improvements

### Others
- Bugfixes in task external events API

## Version 2.0.2, Mon Jun 25, 2018
The 2.0.2 release corresponds to the OmpSs-2 2018.06c release.

### General
- Bugfixes in HWLOC support

## Version 2.0.1, Mon Jun 25, 2018
The 2.0.1 release corresponds to the OmpSs-2 2018.06b release.

### General
- Bugfixes in task reductions

## Version 2.0, Mon Jun 25, 2018
The 2.0 release corresponds to the OmpSs-2 2018.06 release. It introduces support for OmpSs-2@CUDA in Unified Memory NVIDIA devices.
It also supports array task reductions in C/C++ and task priorities. Additionally, it provides two new APIs used by the
[TAMPI](https://github.com/bsc-pm/tampi) library.

### General
- Support for OmpSs-2@CUDA Unified Memory
- Bugfixes and performance improvements

### Dependency System
- Support for array task reductions in C/C++

### Scheduling
- Support for task priorities
- Add priority scheduler

### Others
- Add polling services API
- Add task external events API
- Rename taskloop construct to loop

## Version 1.0.1, Thu Nov 23, 2017
The 1.0.1 release corresponds to the OmpSs-2 2017.11b release.

### General
- Fixes for the building system
- Fixes for the loading system

## Version 1.0, Mon Nov 13, 2017
The 1.0 release corresponds to the OmpSs-2 2017.11 release. It is the first release of the Nanos6 runtime system. It implements the basic
infrastructure to manage the parallelism of user tasks (task creation, task scheduling, etc) and their data dependencies. The task dependency
system supports the nested dependency domain connection, and both early release and weak dependency models.

### General
- General infrastructure of the runtime system
- Support for user tasks and nesting of tasks

### Scheduling
- Implement different schedulers: FIFO, LIFO, etc

### Dependency System
- Implementation of a task data dependency system
- Support for nested dependency domain connection
- Support for early release of task dependencies
- Support for weak task dependencies
- Support for reductions

### Others
- Taskloop construct with dependencies
- Task pause/resume API
