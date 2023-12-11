/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2021-2023 Barcelona Supercomputing Center (BSC)
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
#include "support/config/ConfigVariable.hpp"

#define ALIAS_TRACEPOINT(level, name, str) \
	static void name()\
	{\
		emitGeneric(level, str);\
	}

#define ALIAS_TRACEPOINT_MAYBE(level, name, str) \
	static void name()\
	{\
		if (ovni_thread_isready()) {\
			emitGeneric(level, str);\
		}\
	}

namespace Instrument {
	const ConfigVariable<unsigned int> _level("instrument.ovni.level");

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
		static void emitGeneric(unsigned level, const char *eventCode, Ts... args)
		{
			if (level > _level)
				return;

			struct ovni_ev ev;
			memset(&ev, 0, sizeof(struct ovni_ev));
			ovni_ev_set_clock(&ev, ovni_clock_now());
			ovni_ev_set_mcv(&ev, eventCode);
			addPayload(&ev, args...);
			ovni_ev_emit(&ev);
		}

	public:

		// Nanos6 events divided in categories

		// Scheduler
		ALIAS_TRACEPOINT(2, schedServerEnter, "6S[")
		ALIAS_TRACEPOINT(2, schedServerExit, "6S]")
		ALIAS_TRACEPOINT(2, schedReceiveTask, "6Sr")
		ALIAS_TRACEPOINT(2, schedAssignTask, "6Ss")
		ALIAS_TRACEPOINT(2, schedSelfAssignTask, "6S@")
		ALIAS_TRACEPOINT(3, addReadyTaskEnter, "6Sa")
		ALIAS_TRACEPOINT(3, addReadyTaskExit, "6SA")
		ALIAS_TRACEPOINT(3, processReadyEnter, "6Sp")
		ALIAS_TRACEPOINT(3, processReadyExit, "6SP")
		// Worker
		ALIAS_TRACEPOINT(2, workerLoopEnter, "6W[")
		ALIAS_TRACEPOINT(2, workerLoopExit, "6W]")
		ALIAS_TRACEPOINT(2, handleTaskEnter, "6Wt")
		ALIAS_TRACEPOINT(2, handleTaskExit, "6WT")
		ALIAS_TRACEPOINT(2, switchToEnter, "6Ww")
		ALIAS_TRACEPOINT(2, switchToExit, "6WW")
		ALIAS_TRACEPOINT(2, suspendEnter, "6Ws")
		ALIAS_TRACEPOINT(2, suspendExit, "6WS")
		ALIAS_TRACEPOINT(2, resumeEnter, "6Wr")
		ALIAS_TRACEPOINT(2, resumeExit, "6WR")
		ALIAS_TRACEPOINT(2, spongeModeEnter, "6Wg")
		ALIAS_TRACEPOINT(2, spongeModeExit, "6WG")
		// Worker progressing/resting
		ALIAS_TRACEPOINT(2, workerProgressing, "6Pp")
		ALIAS_TRACEPOINT(2, workerResting, "6Pr")
		ALIAS_TRACEPOINT(2, workerAbsorbing, "6Pa")
		// Submit
		ALIAS_TRACEPOINT(2, submitTaskEnter, "6U[")
		ALIAS_TRACEPOINT(2, submitTaskExit, "6U]")
		// Spawn
		ALIAS_TRACEPOINT(2, spawnFunctionEnter, "6F[")
		ALIAS_TRACEPOINT(2, spawnFunctionExit, "6F]")
		// Dependencies
		ALIAS_TRACEPOINT(2, registerAccessesEnter, "6Dr")
		ALIAS_TRACEPOINT(2, registerAccessesExit, "6DR")
		ALIAS_TRACEPOINT(2, unregisterAccessesEnter, "6Du")
		ALIAS_TRACEPOINT(2, unregisterAccessesExit, "6DU")
		// Memory
		// Cannot ensure the ovni thread is initialized because the
		// memory allocator can be called by the loader when creating
		// space for static variables, before Nanos6 initializes the
		// instrumentation.
		ALIAS_TRACEPOINT_MAYBE(3, memoryAllocEnter, "6Ma")
		ALIAS_TRACEPOINT_MAYBE(3, memoryAllocExit, "6MA")
		ALIAS_TRACEPOINT_MAYBE(3, memoryFreeEnter, "6Mf")
		ALIAS_TRACEPOINT_MAYBE(3, memoryFreeExit, "6MF")
		// Blocking
		ALIAS_TRACEPOINT(2, blockEnter, "6Bb")
		ALIAS_TRACEPOINT(2, blockExit, "6BB")
		ALIAS_TRACEPOINT(2, unblockEnter, "6Bu")
		ALIAS_TRACEPOINT(2, unblockExit, "6BU")
		ALIAS_TRACEPOINT(2, taskWaitEnter, "6Bw")
		ALIAS_TRACEPOINT(2, taskWaitExit, "6BW")
		ALIAS_TRACEPOINT(2, waitForEnter, "6Bf")
		ALIAS_TRACEPOINT(2, waitForExit, "6BF")
		// Task creation
		ALIAS_TRACEPOINT(2, enterCreateTask, "6C[")
		ALIAS_TRACEPOINT(2, exitCreateTask, "6C]")
		// Task body
		ALIAS_TRACEPOINT(2, taskBodyEnter, "6t[")
		ALIAS_TRACEPOINT(2, taskBodyExit, "6t]")

		// Task lifecycle (these track the state of tasks)
		static void taskCreate(uint32_t taskId, uint32_t typeId)
		{
			emitGeneric(1, "6Tc", taskId, typeId);
		}

		static void taskExecute(uint32_t taskId)
		{
			emitGeneric(1, "6Tx", taskId);
		}

		static void taskPause(uint32_t taskId)
		{
			emitGeneric(1, "6Tp", taskId);
		}

		static void taskResume(uint32_t taskId)
		{
			emitGeneric(1, "6Tr", taskId);
		}

		static void taskEnd(uint32_t taskId)
		{
			emitGeneric(1, "6Te", taskId);
		}

		// Large things like strings need to be sent using jumbo events
		static inline void typeCreate(uint32_t typeId, const char *label)
		{
			if (_level >= 1) {
				OvniJumboEvent event;
				event.addScalarPayload(typeId);
				event.addString(label);
				event.emit("6Yc");
			}
		}

		// Generic ovni events
		ALIAS_TRACEPOINT(1, burst, "OB.")
		ALIAS_TRACEPOINT(1, threadPause, "OHp")
		ALIAS_TRACEPOINT(1, threadResume, "OHr")
		ALIAS_TRACEPOINT(1, threadCool, "OHc")
		ALIAS_TRACEPOINT(1, threadWarm, "OHw")

		static void affinitySet(int32_t cpu)
		{
			emitGeneric(1, "OAs", cpu);
		}

		static void affinityRemote(int32_t cpu, int32_t tid)
		{
			emitGeneric(1, "OAr", cpu, tid);
		}

		static void cpuCount(int32_t count, int32_t maxcpu)
		{
			emitGeneric(1, "OCn", count, maxcpu);
		}

		static void threadCreate(int32_t cpu, uint64_t tag)
		{
			emitGeneric(1, "OHC", cpu, tag);
		}

		static void threadExecute(int32_t cpu, int32_t creatorTid, uint64_t tag)
		{
			emitGeneric(1, "OHx", cpu, creatorTid, tag);
		}

		static void threadTypeBegin(char type)
		{
			// 6HW 6HL 6HM
			char mcv[] = {'6', 'H', '?', '\0'};
			mcv[2] = tolower(type);
			emitGeneric(1, mcv);
		}

		static void threadTypeEnd(char type)
		{
			// 6Hw 6Hl 6Hm
			char mcv[] = {'6', 'H', '?', '\0'};
			mcv[2] = toupper(type);
			emitGeneric(1, mcv);
		}

		static void addCPU(int index, int phyid)
		{
			ovni_add_cpu(index, phyid);
		}

		static void threadEnd()
		{
			emitGeneric(1, "OHe");
			// Flush the events to disk before killing the thread
			ovni_flush();
		}

		static void threadSignal(int32_t tid)
		{
			emitGeneric(2, "6W*", tid);
		}

		static void checkVersion()
		{
			// Make sure the current libovni version loaded at run-time is compatible
			// with the libovni that the runtime was compiled against. This should be
			// the first call to ovni
			ovni_version_check();
		}

		static void procInit()
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

			// Initialize ovni with APPID = 1, as there is only one application in this runtime
			ovni_proc_init(1, loomName, pid);
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
			ovni_thread_require("nanos6", "1.0.0");
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
