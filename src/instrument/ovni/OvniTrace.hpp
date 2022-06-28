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

#include "lowlevel/CompatSyscalls.hpp"
#include "lowlevel/FatalErrorHandler.hpp"

#define ALIAS_TRACEPOINT(name, str) \
	static void name()              \
	{                               \
		emitGeneric(str);           \
	}

namespace Instrument {

	class OvniJumboEvent {
		static constexpr size_t _jumboBufferSize = 1024;
		uint8_t _buffer[_jumboBufferSize];
		size_t _currentOffset;
		size_t _sizeLeft;

	public:
		OvniJumboEvent() :
			_currentOffset(0),
			_sizeLeft(_jumboBufferSize)
		{
		}

		void emit(const char *mcv)
		{
			struct ovni_ev ev;
			memset(&ev, 0, sizeof(struct ovni_ev));
			ovni_ev_set_clock(&ev, ovni_clock_now());
			ovni_ev_set_mcv(&ev, mcv);
			ovni_ev_jumbo_emit(&ev, _buffer, _jumboBufferSize - _sizeLeft);
		}

		template<typename T>
		void addScalarPayload(T payload)
		{
			// We cannot safely truncate these payloads
			assert(_sizeLeft >= sizeof(T));

			memcpy(&_buffer[_currentOffset], &payload, sizeof(T));
			_currentOffset += sizeof(T);
			_sizeLeft -= sizeof(T);
		}

		void addString(const char *str)
		{
			// NULL strings translated as empty
			if (!str)
				str = "";

			// Copy everything except the null-termination
			size_t len = std::min(strlen(str), _sizeLeft - 1);
			memcpy(&_buffer[_currentOffset], str, len);

			_currentOffset += len;
			_sizeLeft -= len;

			// NULL-terminate the string
			assert(_sizeLeft >= 1);
			_buffer[_currentOffset++] = '\0';
			_sizeLeft--;
		}
	};

	class Ovni {
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
			struct ovni_ev ev;
			memset(&ev, 0, sizeof(struct ovni_ev));
			ovni_ev_set_clock(&ev, ovni_clock_now());
			ovni_ev_set_mcv(&ev, eventCode);
			addPayload(&ev, args...);
			ovni_ev_emit(&ev);
		}

	public:
		// Nanos6 events
		ALIAS_TRACEPOINT(schedReceiveTask, "6Sr")
		ALIAS_TRACEPOINT(schedAssignTask, "6Ss")
		ALIAS_TRACEPOINT(schedSelfAssignTask, "6S@")
		ALIAS_TRACEPOINT(schedHungry, "6Sh")
		ALIAS_TRACEPOINT(schedFill, "6Sf")
		ALIAS_TRACEPOINT(schedServerEnter, "6S[")
		ALIAS_TRACEPOINT(schedServerExit, "6S]")
		ALIAS_TRACEPOINT(schedSubmitEnter, "6Su")
		ALIAS_TRACEPOINT(schedSubmitExit, "6SU")
		ALIAS_TRACEPOINT(enterSubmitTask, "6U[")
		ALIAS_TRACEPOINT(exitSubmitTask, "6U]")
		ALIAS_TRACEPOINT(blockEnter, "6Bb")
		ALIAS_TRACEPOINT(blockExit, "6BB")
		ALIAS_TRACEPOINT(waitforEnter, "6Bw")
		ALIAS_TRACEPOINT(waitforExit, "6BW")
		ALIAS_TRACEPOINT(registerAccessesEnter, "6Dr")
		ALIAS_TRACEPOINT(registerAccessesExit, "6DR")
		ALIAS_TRACEPOINT(unregisterAccessesEnter, "6Du")
		ALIAS_TRACEPOINT(unregisterAccessesExit, "6DU")
		ALIAS_TRACEPOINT(taskWaitEnter, "6Wt")
		ALIAS_TRACEPOINT(taskWaitExit, "6WT")
		ALIAS_TRACEPOINT(exitCreateTask, "6TC")
		ALIAS_TRACEPOINT(spawnFunctionEnter, "6Hs")
		ALIAS_TRACEPOINT(spawnFunctionExit, "6HS")

		static void taskCreate(uint32_t taskId, uint32_t typeId)
		{
			emitGeneric("6Tc", taskId, typeId);
		}

		static void taskExecute(uint32_t taskId)
		{
			emitGeneric("6Tx", taskId);
		}

		static void taskPause(uint32_t taskId)
		{
			emitGeneric("6Tp", taskId);
		}

		static void taskResume(uint32_t taskId)
		{
			emitGeneric("6Tr", taskId);
		}

		static void taskEnd(uint32_t taskId)
		{
			emitGeneric("6Te", taskId);
		}

		static void unblockEnter(uint32_t taskId)
		{
			emitGeneric("6Bu", taskId);
		}

		static void unblockExit(uint32_t taskId)
		{
			emitGeneric("6BU", taskId);
		}

		// Large things like strings need to be sent using jumbo events
		static inline void typeCreate(uint32_t typeId, const char *label)
		{
			OvniJumboEvent event;
			event.addScalarPayload(typeId);
			event.addString(label);
			event.emit("6Yc");
		}

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

		static void addCPU(int index, int phyid)
		{
			ovni_add_cpu(index, phyid);
		}

		static void threadEnd()
		{
			emitGeneric("OHe");
			// Flush the events to disk before killing the thread
			ovni_flush();
		}

		static void initialize()
		{
			char hostname[HOST_NAME_MAX + 1];
			char loomName[HOST_NAME_MAX + 64];

			if (gethostname(hostname, HOST_NAME_MAX) != 0) {
				FatalErrorHandler::fail("Could not get hostname while initializing instrumentation");
			}

			// gethostname() may not null-terminate the buffer
			hostname[HOST_NAME_MAX] = '\0';

			pid_t pid = getpid();
			sprintf(loomName, "%s.%d", hostname, pid);

			// Initialize OVNI with APPID = 1, as there is only one application in this runtime
			ovni_proc_init(1, loomName, pid);
		}

		static void finalize()
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

			emitGeneric("6Ha");
		}

		static void threadDetach()
		{
			emitGeneric("6HA");
			// Flush the events to disk before detaching the thread
			ovni_flush();
		}

		static void threadMaybeInit()
		{
			if (!ovni_thread_isready()) {
				threadInit();
				threadExecute(-1, -1, -1);
			}
		}

		static void flush()
		{
			ovni_flush();
		}
	};

} // namespace Instrument

#endif // OVNI_TRACE_HPP
