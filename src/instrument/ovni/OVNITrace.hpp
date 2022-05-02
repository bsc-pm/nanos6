/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2021-2022 Barcelona Supercomputing Center (BSC)
*/

#ifndef OVNI_TRACE_HPP
#define OVNI_TRACE_HPP

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <ovni.h>
#include <string>
#include <sys/syscall.h>
#include <unistd.h>

#include "lowlevel/FatalErrorHandler.hpp"

// Define gettid for older glibc versions (below 2.30)
#if !__GLIBC_PREREQ(2, 30)
static inline pid_t gettid(void)
{
	return (pid_t)syscall(SYS_gettid);
}
#endif

#define ALIAS_TRACEPOINT(name, str) \
	static void name()              \
	{                               \
		emitGeneric(str);           \
	}

namespace Instrument {

	class OVNIJumboEvent {
		static constexpr size_t jumboBufferSize = 1024;
		uint8_t buffer[jumboBufferSize];
		size_t currentOffset = 0;
		size_t sizeLeft = jumboBufferSize;

	public:
		void emit(const char *mcv)
		{
			struct ovni_ev ev = {};
			memset(&ev, 0, sizeof(struct ovni_ev));
			ovni_ev_set_clock(&ev, ovni_clock_now());
			ovni_ev_set_mcv(&ev, mcv);
			ovni_ev_jumbo_emit(&ev, buffer, jumboBufferSize - sizeLeft);
		}

		template<typename T>
		void addScalarPayload(T payload)
		{
			// We cannot safely truncate these payloads
			assert(sizeLeft >= sizeof(T));

			memcpy(&buffer[currentOffset], &payload, sizeof(T));
			currentOffset += sizeof(T);
			sizeLeft -= sizeof(T);
		}

		void addString(const char *str)
		{
			// NULL strings translated as empty
			if (!str)
				str = "";

			// Copy everything except the null-termination
			size_t len = std::min(strlen(str), sizeLeft - 1);
			memcpy(&buffer[currentOffset], str, len);

			currentOffset += len;
			sizeLeft -= len;

			// NULL-terminate the string
			assert(sizeLeft >= 1);
			buffer[currentOffset++] = '\0';
			sizeLeft--;
		}
	};

	class OVNI {
		template <typename T, typename... Ts>
		static void addPayload(ovni_ev *ev, T first, Ts... args)
		{
			ovni_payload_add(ev, (uint8_t *)&first, sizeof(T));
			addPayload(ev, args...);
		}

		static void addPayload(__attribute__((unused)) ovni_ev *ev)
		{
		}

		template <typename... Ts>
		static void emitGeneric(const char *eventCode, Ts... args)
		{
			struct ovni_ev ev = {};
			memset(&ev, 0, sizeof(struct ovni_ev));
			ovni_ev_set_clock(&ev, ovni_clock_now());
			ovni_ev_set_mcv(&ev, eventCode);
			addPayload(&ev, args...);
			ovni_ev_emit(&ev);
		}

	public:
		// nOS-V events
		ALIAS_TRACEPOINT(workerEnter, "VHw")
		ALIAS_TRACEPOINT(workerExit, "VHW")
		ALIAS_TRACEPOINT(delegateEnter, "VHd")
		ALIAS_TRACEPOINT(delegateExit, "VHD")
		ALIAS_TRACEPOINT(schedRecv, "VSr")
		ALIAS_TRACEPOINT(schedSend, "VSs")
		ALIAS_TRACEPOINT(schedSelfAssign, "VS@")
		ALIAS_TRACEPOINT(schedHungry, "VSh")
		ALIAS_TRACEPOINT(schedFill, "VSf")
		ALIAS_TRACEPOINT(schedServerEnter, "VS[")
		ALIAS_TRACEPOINT(schedServerExit, "VS]")
		ALIAS_TRACEPOINT(schedSubmitEnter, "VU[")
		ALIAS_TRACEPOINT(schedSubmitExit, "VU]")
		ALIAS_TRACEPOINT(sallocEnter, "VMa")
		ALIAS_TRACEPOINT(sallocExit, "VMA")
		ALIAS_TRACEPOINT(sfreeEnter, "VMf")
		ALIAS_TRACEPOINT(sfreeExit, "VMF")
		ALIAS_TRACEPOINT(submitEnter, "VAs")
		ALIAS_TRACEPOINT(submitExit, "VAS")
		ALIAS_TRACEPOINT(pauseEnter, "VAp")
		ALIAS_TRACEPOINT(pauseExit, "VAP")
		ALIAS_TRACEPOINT(yieldEnter, "VAy")
		ALIAS_TRACEPOINT(yieldExit, "VAY")
		ALIAS_TRACEPOINT(waitforEnter, "VAw")
		ALIAS_TRACEPOINT(waitforExit, "VAW")
		ALIAS_TRACEPOINT(schedpointEnter, "VAc")
		ALIAS_TRACEPOINT(schedpointExit, "VAC")

		// nOS-V events
		static void taskCreate(uint32_t taskId, uint32_t typeId)
		{
			emitGeneric("VTc", taskId, typeId);
		}

		static void taskExecute(uint32_t taskId)
		{
			emitGeneric("VTx", taskId);
		}

		static void taskPause(uint32_t taskId)
		{
			emitGeneric("VTp", taskId);
		}

		static void taskResume(uint32_t taskId)
		{
			emitGeneric("VTr", taskId);
		}

		static void taskEnd(uint32_t taskId)
		{
			emitGeneric("VTe", taskId);
		}

		// Large things like strings need to be sent using jumbo events
		static inline void typeCreate(uint32_t typeId, const char *label)
		{
			OVNIJumboEvent event;
			event.addScalarPayload(typeId);
			event.addString(label);
			event.emit("VYc");
		}

		// NODES events
		ALIAS_TRACEPOINT(registerAccessesEnter, "DR[")
		ALIAS_TRACEPOINT(registerAccessesExit, "DR]")
		ALIAS_TRACEPOINT(unregisterAccessesEnter, "DU[")
		ALIAS_TRACEPOINT(unregisterAccessesExit, "DU]")
		ALIAS_TRACEPOINT(waitIf0Enter, "DW[")
		ALIAS_TRACEPOINT(waitIf0Exit, "DW]")
		ALIAS_TRACEPOINT(inlineIf0Enter, "DI[")
		ALIAS_TRACEPOINT(inlineIf0Exit, "DI]")
		ALIAS_TRACEPOINT(taskWaitEnter, "DT[")
		ALIAS_TRACEPOINT(taskWaitExit, "DT]")
		ALIAS_TRACEPOINT(createTaskEnter, "DC[")
		ALIAS_TRACEPOINT(createTaskExit, "DC]")
		ALIAS_TRACEPOINT(submitTaskEnter, "DS[")
		ALIAS_TRACEPOINT(submitTaskExit, "DS]")
		ALIAS_TRACEPOINT(spawnFunctionEnter, "DP[")
		ALIAS_TRACEPOINT(spawnFunctionExit, "DP]")

		// Generic OVNI events
		ALIAS_TRACEPOINT(burst, "OB.")
		ALIAS_TRACEPOINT(threadPause, "OHp")
		ALIAS_TRACEPOINT(threadResume, "OHr")
		ALIAS_TRACEPOINT(threadCool, "OHc")
		ALIAS_TRACEPOINT(threadWarm, "OHw")

		static void affinitySet(int32_t cpu)
		{
			emitGeneric("OAs", cpu);
		}

		static void affinityRemote(int32_t cpu, int32_t tid)
		{
			emitGeneric("OAr", cpu, tid);
		}

		static void cpuCount(int32_t count, int32_t maxcpu)
		{
			emitGeneric("OCn", count, maxcpu);
		}

		static void threadCreate(int32_t cpu, uint64_t tag)
		{
			emitGeneric("OHC", cpu, tag);
		}

		static void threadExecute(int32_t cpu, int32_t creatorTid, uint64_t tag)
		{
			emitGeneric("OHx", cpu, creatorTid, tag);
		}

		static void cpuId(int index, int phyid)
		{
			ovni_add_cpu(index, phyid);
		}

		static void threadEnd()
		{
			emitGeneric("OHe");
			// Flush the events to disk before killing the thread
			ovni_flush();
		}

		static void procInit()
		{
			char hostname[HOST_NAME_MAX + 1];

			if (gethostname(hostname, HOST_NAME_MAX) != 0) {
				FatalErrorHandler::fail("Could not get hostname while initializing instrumentation");
			}

			// gethostname() may not null-terminate the buffer
			hostname[HOST_NAME_MAX] = '\0';

			// Initialize OVNI with APPID = 1, as there is only one application in this runtime
			ovni_proc_init(1, hostname, getpid());
		}

		static void procFini()
		{
			ovni_proc_fini();
		}

		static void genBursts()
		{
			for (int i = 0; i < 100; i++)
				burst();
		}

		static void threadInit()
		{
			ovni_thread_init(gettid());
		}

		static void threadAttach()
		{
			if (!ovni_thread_isready())
				FatalErrorHandler::fail("The current thread is not instrumented correctly");

			emitGeneric("VHa");
		}

		static void threadDetach()
		{
			emitGeneric("VHA");
			// Flush the events to disk before detaching the thread
			ovni_flush();
		}
	};

} // namespace Instrument

#endif // OVNI_TRACE_HPP
