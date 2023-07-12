# Using Device tasks

Nanos6 is designed with the capacity to be extended to handle tasks running on
heterogeneous systems, using Accelerators.

For this purpose, a `device` clause specifying the accelerator type can be used in a
task directive, to annotate code written for an Accelerator so that the runtime handles
it properly.

Device tasks should be written as *outlined tasks*; this means that the whole
task code is encapsulated in a dedicated function. Then the **prototype** of this
function is annotated with an `oss task` directive.

Example:

*In a header file:*

```c
#pragma oss task device(<type>) in(...) out(...) ... /* rest of clauses */
void function_foo(<params>);
```

*In a code file:*

```c
void function_foo(<params>)
{
	/* function code */
}
```

This will result in every call to `function_foo()` in the program to implicitly
create and launch a new task.

# Supported Devices

Currently, the supported devices in Nanos6 are:

1. [CUDA](CUDA.md)
1. [OpenACC](OpenACC.md)

To use device tasks in Nanos6, the user must explicitly enable the specific devices they intend
to use, during configuration as described in [README](README.md)

For device-specific information about usage and compilation, refer to the respective guides.
