#
#	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
#
#	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
#

from abc import ABC
from runtime import RuntimeModel
from paravertrace import ParaverTrace, ExtraeEventTypes, ExtraeEventCollection

class RuntimeActivity:
	Runtime     = 1
	BusyWaiting = 2
	Task        = 3

class ParaverView(ABC):
	def __init__(self):
		self._hooks = None

	def hooks(self):
		return self._hooks

	def start(self, payload):
		""" called before any CTF event is processed """
		pass

class ParaverViewRuntimeCode(ParaverView):
	def __init__(self):
		super().__init__()
		self._hooks = [
			("nanos6:thread_resume",            self.hook_threadResume),
			("nanos6:external_thread_resume",   self.hook_threadResume),
			("nanos6:thread_suspend",           self.hook_threadStop),
			("nanos6:external_thread_suspend",  self.hook_threadStop),
			("nanos6:thread_shutdown",          self.hook_threadStop),
			("nanos6:external_thread_shutdown", self.hook_threadStop),
		]
		ParaverTrace.addEventTypeAndValue(ExtraeEventTypes.RUNTIME_CODE, {RuntimeActivity.Runtime: "Runtime"}, "Runtime: Runtime Code")

	def hook_threadResume(self, _, payload):
		payload.append((ExtraeEventTypes.RUNTIME_CODE, RuntimeActivity.Runtime))

	def hook_threadStop(self, _, payload):
		payload.append((ExtraeEventTypes.RUNTIME_CODE, 0))

class ParaverViewRuntimeBusyWaiting(ParaverView):
	def __init__(self):
		super().__init__()
		self._hooks = [
			("nanos6:thread_shutdown",        self.hook_threadShutdown),
			("nanos6:worker_enter_busy_wait", self.hook_enterBusyWait),
			("nanos6:worker_exit_busy_wait",  self.hook_exitBusyWait)
		]
		ParaverTrace.addEventTypeAndValue(ExtraeEventTypes.RUNTIME_BUSYWAITING, {RuntimeActivity.BusyWaiting: "BusyWait"}, "Runtime: Busy Waiting")

	def hook_threadShutdown(self, _, payload):
		# In case we were busy waiting, we emit another "0" event to unstack
		# it. We might emit the event event if the thread was not busy waiting,
		# but its cheaper to always do it rather than keeping track of it's
		# status. Paraver just ignores extra "unstack" events.
		payload.append((ExtraeEventTypes.RUNTIME_BUSYWAITING, 0))

	def hook_enterBusyWait(self, _, payload):
		payload.append((ExtraeEventTypes.RUNTIME_BUSYWAITING, RuntimeActivity.BusyWaiting))

	def hook_exitBusyWait(self, _, payload):
		payload.append((ExtraeEventTypes.RUNTIME_BUSYWAITING, 0))

class ParaverViewRuntimeTasks(ParaverView):
	def __init__(self):
		super().__init__()
		self._hooks = [
			("nanos6:task_execute", self.hook_taskExecute),
			("nanos6:task_block",   self.hook_taskStop),
			("nanos6:task_end",     self.hook_taskStop)
		]
		ParaverTrace.addEventTypeAndValue(ExtraeEventTypes.RUNTIME_TASKS, {RuntimeActivity.Task: "Task"}, "Runtime: Task Code")

	def hook_taskExecute(self, _, payload):
		payload.append((ExtraeEventTypes.RUNTIME_TASKS, RuntimeActivity.Task))

	def hook_taskStop(self, _, payload):
		payload.append((ExtraeEventTypes.RUNTIME_TASKS, 0))

class ParaverViewTaskLabel(ParaverView):
	def __init__(self):
		super().__init__()
		self._hooks = [
			("nanos6:task_label",   self.hook_taskLabel),
			("nanos6:task_execute", self.hook_taskExecute),
			("nanos6:task_block",   self.hook_taskStop),
			("nanos6:task_end",     self.hook_taskStop)
		]
		ParaverTrace.addEventType(ExtraeEventTypes.RUNNING_TASK_LABEL, "Running Task Label")

	def hook_taskLabel(self, event, _):
		label      = event["label"]
		taskTypeID = event["type"]
		ParaverTrace.addEventTypeAndValue(ExtraeEventTypes.RUNNING_TASK_LABEL, {taskTypeID : label})

	def hook_taskExecute(self, event, payload):
		taskId = event["id"]
		taskTypeId = RuntimeModel.getTaskTypeId(taskId)
		payload.append((ExtraeEventTypes.RUNNING_TASK_LABEL, taskTypeId))

	def hook_taskStop(self, event, payload):
		taskId = event["id"]
		taskTypeId = RuntimeModel.getTaskTypeId(taskId)
		payload.append((ExtraeEventTypes.RUNNING_TASK_LABEL, 0))

class ParaverViewTaskSource(ParaverView):
	def __init__(self):
		super().__init__()
		self._hooks = [
			("nanos6:task_label",   self.hook_taskLabel),
			("nanos6:task_execute", self.hook_taskExecute),
			("nanos6:task_block",   self.hook_taskStop),
			("nanos6:task_end",     self.hook_taskStop)
		]
		ParaverTrace.addEventType(ExtraeEventTypes.RUNNING_TASK_SOURCE, "Running Task Source")

	def hook_taskLabel(self, event, _):
		source     = event["source"]
		taskTypeID = event["type"]
		ParaverTrace.addEventTypeAndValue(ExtraeEventTypes.RUNNING_TASK_SOURCE, {taskTypeID : source})

	def hook_taskExecute(self, event, payload):
		taskId = event["id"]
		taskTypeId = RuntimeModel.getTaskTypeId(taskId)
		payload.append((ExtraeEventTypes.RUNNING_TASK_SOURCE, taskTypeId))

	def hook_taskStop(self, event, payload):
		taskId = event["id"]
		taskTypeId = RuntimeModel.getTaskTypeId(taskId)
		payload.append((ExtraeEventTypes.RUNNING_TASK_SOURCE, 0))


class ParaverViewHardwareCounters(ParaverView):
	def __init__(self):
		super().__init__()
		self._hooks = [
			("nanos6:thread_suspend",  self.hook_getHardwareCounters),
			("nanos6:thread_shutdown", self.hook_getHardwareCounters),
			("nanos6:task_execute",    self.hook_getHardwareCounters),
			("nanos6:task_block",      self.hook_getHardwareCounters),
			("nanos6:task_end",        self.hook_getHardwareCounters)
		]

		self._eventsHardwareCounters = ExtraeEventCollection(42010000, 7)
		self._eventsHardwareCounters.addEvents([
			("PAPI_TOT_INS", 42000050, "PAPI_TOT_INS [Instr completed]"),
			("PAPI_TOT_CYC", 42000059, "PAPI_TOT_CYC [Total cycles]"),
			("PAPI_L1_DCM",  42000000, "PAPI_L1_DCM  [L1D cache misses]"),
			("PAPI_L2_DCM",  42000002, "PAPI_L2_DCM  [L2D cache misses]"),
			("PAPI_L3_TCM",  42000008, "PAPI_L3_TCM  [L3 cache misses]"),
			("PAPI_BR_INS",  42000055, "PAPI_BR_INS  [Branches]"),
			("PAPI_BR_MSP",  42000046, "PAPI_BR_MSP  [Cond br mspredictd]")
		])
		ParaverTrace.addEventCollection(self._eventsHardwareCounters)

	def getHardwareCountersId(self, name):
		try:
			extraeId = self._eventsHardwareCounters.getExtraeId(name)
		except:
			extraeId = self._eventsHardwareCounters.addUnknownEvent(name)
			print("Warning: Missing Hardware Counter Id for " + name + " assigning the temporal id " + str(extraeId))
		return extraeId

	def hook_getHardwareCounters(self, event, payload):
		# babeltrace does not allow to check for a field like this
		# if not "hwc" in event:
		# 	return
		try:
			hwc = event["hwc"]
		except:
			return
		# we use the CTF field name to obtain the right Extrae ID
		for key, val in hwc.items():
			extraeId = self.getHardwareCountersId(key)
			payload.append((extraeId, val))

class ParaverViewThreadId(ParaverView):
	def __init__(self):
		super().__init__()
		self._hooks = [
			("nanos6:external_thread_create",   self.hook_threadCreate),
			("nanos6:thread_create",            self.hook_threadCreate),
			("nanos6:thread_resume",            self.hook_threadResume),
			("nanos6:external_thread_resume",   self.hook_threadResume),
			("nanos6:thread_suspend",           self.hook_threadSuspend),
			("nanos6:external_thread_suspend",  self.hook_threadSuspend),
			("nanos6:thread_shutdown",          self.hook_threadSuspend),
			("nanos6:external_thread_shutdown", self.hook_threadSuspend),
		]
		self._colorMap = {}
		self._colorMap_indx = 1
		ParaverTrace.addEventType(ExtraeEventTypes.RUNNING_THREAD_TID, "Worker Thread Id (TID)")

	def hook_threadCreate(self, event, payload):
		tid = event["tid"]
		assert(not tid in self._colorMap.keys())
		color = self._colorMap_indx
		self._colorMap[tid] = color
		self._colorMap_indx += 1
		ParaverTrace.addEventTypeAndValue(ExtraeEventTypes.RUNNING_THREAD_TID, {color: "thread " + str(tid)})

	def hook_threadResume(self, event, payload):
		tid = event["tid"]
		color = self._colorMap[tid]
		payload.append((ExtraeEventTypes.RUNNING_THREAD_TID, color))

	def hook_threadSuspend(self, event, payload):
		payload.append((ExtraeEventTypes.RUNNING_THREAD_TID, 0))

class ParaverViewTaskId(ParaverView):
	def __init__(self):
		super().__init__()
		self._hooks = [
			("nanos6:task_execute",   self.hook_taskExecute),
			("nanos6:task_block",     self.hook_taskStop),
			("nanos6:task_end",       self.hook_taskStop)
		]
		ParaverTrace.addEventType(ExtraeEventTypes.RUNNING_TASK_ID, "Task ID")

	def hook_taskExecute(self, event, payload):
		taskId = event["id"]
		payload.append((ExtraeEventTypes.RUNNING_TASK_ID, taskId))

	def hook_taskStop(self, event, payload):
		payload.append((ExtraeEventTypes.RUNNING_TASK_ID, 0))



class ParaverViewRuntimeSubsystems(ParaverView):
	class Status:
		Idle                 = 0
		Runtime              = 1
		BusyWait             = 2
		Task                 = 3
		DependencyRegister   = 4
		DependencyUnregister = 5
		SchedulerAddTask     = 6
		SchedulerGetTask     = 7
		TaskCreate           = 8
		TaskArgsInit         = 9
		TaskSubmit           = 10
		TaskforInit          = 11
		Debug                = 100

	def stackEvent(func):
		def wrapper(self, event, payload):
			stack = self.getEventStack(event)
			val = func(self, event)
			stack.append(val)
			self.emit(stack, payload)
		return wrapper

	def unstackAndStackEvent(func):
		def wrapper(self, event, payload):
			stack = self.getEventStack(event)
			stack.pop()
			val = func(self, event)
			stack.append(val)
			self.emit(stack, payload)
		return wrapper

	def __init__(self):
		super().__init__()
		self._hooks = [
			("nanos6:thread_create",               self.hook_initStack),
			("nanos6:external_thread_create",      self.hook_initStack),
			("nanos6:thread_resume",               self.hook_eventContinue),
			("nanos6:external_thread_resume",      self.hook_eventContinue),
			("nanos6:thread_suspend",              self.hook_eventStop),
			("nanos6:external_thread_suspend",     self.hook_eventStop),
			("nanos6:thread_shutdown",             self.hook_eventStop),
			("nanos6:external_thread_shutdown",    self.hook_eventStop),
			("nanos6:task_execute",                self.hook_task),
			("nanos6:task_block",                  self.hook_unstack),
			("nanos6:task_end",                    self.hook_unstack),
			("nanos6:worker_enter_busy_wait",      self.hook_busyWait),
			("nanos6:worker_exit_busy_wait",       self.hook_unstack),
			("nanos6:dependency_register_enter",   self.hook_dependencyRegister),
			("nanos6:dependency_register_exit",    self.hook_unstack),
			("nanos6:dependency_unregister_enter", self.hook_dependencyUnregister),
			("nanos6:dependency_unregister_exit",  self.hook_unstack),
			("nanos6:scheduler_add_task_enter",    self.hook_schedulerAddTask),
			("nanos6:scheduler_add_task_exit",     self.hook_unstack),
			("nanos6:scheduler_get_task_enter",    self.hook_schedulerGetTask),
			("nanos6:scheduler_get_task_exit",     self.hook_unstack),
			("nanos6:task_create_enter",           self.hook_taskCreate),
			("nanos6:task_create_exit",            self.hook_taskBetweenCreateAndSubmit),
			("nanos6:task_submit_enter",           self.hook_taskSubmit),
			("nanos6:task_submit_exit",            self.hook_unstack),
			("nanos6:taskfor_init_enter",          self.hook_taskforInit),
			("nanos6:taskfor_init_exit",           self.hook_unstack),
			("nanos6:debug_register",              self.hook_debugRegister),
			("nanos6:debug_enter",                 self.hook_debug),
			("nanos6:debug_exit",                  self.hook_unstack),
		]
		status = {
			self.Status.Idle:                 "Idle",
			self.Status.Runtime:              "Runtime",
			self.Status.BusyWait:             "Busy Wait",
			self.Status.Task:                 "Task",
			self.Status.DependencyRegister:   "Dependency: Register",
			self.Status.DependencyUnregister: "Dependency: Unregister",
			self.Status.SchedulerAddTask:     "Scheduler: Add Ready Task",
			self.Status.SchedulerGetTask:     "Scheduler: Get Ready Task",
			self.Status.TaskCreate:           "Task: Create",
			self.Status.TaskArgsInit:         "Task: Arguments Init",
			self.Status.TaskSubmit:           "Task: Submit",
			self.Status.TaskforInit:          "Task: Taskfor Collaborator Init",
		}
		ParaverTrace.addEventTypeAndValue(ExtraeEventTypes.RUNTIME_SUBSYSTEMS, status, "Runtime Subsystems")

	def hook_initStack(self, event, payload):
		tid = event["tid"]
		thread = RuntimeModel.getThread(tid)
		thread.eventStack = [self.Status.Runtime]

	def hook_unstack(self, event, payload):
		stack = self.getEventStack(event)
		stack.pop()
		self.emit(stack, payload)

	def hook_eventContinue(self, event, payload):
		stack = self.getEventStack(event)
		self.emit(stack, payload)

	def hook_eventStop(self, _, payload):
		self.emitVal(self.Status.Idle, payload)

	@stackEvent
	def hook_task(self, _):
		return self.Status.Task

	@stackEvent
	def hook_busyWait(self, _):
		return self.Status.BusyWait

	@stackEvent
	def hook_dependencyRegister(self, _):
		return self.Status.DependencyRegister

	@stackEvent
	def hook_dependencyUnregister(self, _):
		return self.Status.DependencyUnregister

	@stackEvent
	def hook_schedulerAddTask(self, _):
		return self.Status.SchedulerAddTask

	@stackEvent
	def hook_schedulerGetTask(self, _):
		return self.Status.SchedulerGetTask

	@stackEvent
	def hook_taskCreate(self, _):
		return self.Status.TaskCreate

	@unstackAndStackEvent
	def hook_taskBetweenCreateAndSubmit(self, _):
		return self.Status.TaskArgsInit

	@unstackAndStackEvent
	def hook_taskSubmit(self, _):
		return self.Status.TaskSubmit

	@stackEvent
	def hook_taskforInit(self, _):
		return self.Status.TaskforInit

	@stackEvent
	def hook_debug(self, event):
		debugId = event["id"]
		return self.Status.Debug + debugId

	def hook_debugRegister(self, event, payload):
		name    = event["name"]
		debugId = event["id"]

		extraeId   = self.Status.Debug + debugId
		extraeName = "Debug: " + str(name)
		ParaverTrace.addEventTypeAndValue(ExtraeEventTypes.RUNTIME_SUBSYSTEMS, {extraeId : extraeName})

	def getEventStack(self, event):
		thread = RuntimeModel.getCurrentThread(event)
		stack = thread.eventStack
		assert(stack != None)
		return stack

	def emit(self, stack, payload):
		assert(len(stack) > 0)
		payload.append((ExtraeEventTypes.RUNTIME_SUBSYSTEMS, stack[-1]))

	def emitVal(self, val, payload):
		payload.append((ExtraeEventTypes.RUNTIME_SUBSYSTEMS, val))

#class ParaverViewReadyTasks(ParaverView):
#	def __init__(self):
#		super().__init__()
#		self._hooks = [
#			("nanos6:scheduler_add_task_enter", self.hook_schedulerAddTask),
#			("nanos6:scheduler_get_task_exit", self.hook_schedulerGetTask),
#		]
#		ParaverTrace.addEventType(ExtraeEventTypes.NUMBER_OF_READY_TASKS, "Number of Ready Tasks")
#		self.readyTasksCount = 0
#
#	def start(self, payload):
#		payload.append((ExtraeEventTypes.NUMBER_OF_READY_TASKS, 0))
#
#	def hook_schedulerAddTask(self, _, payload):
#		self.readyTasksCount += 1
#		#print("add: " + str(self.readyTasksCount))
#		payload.append((ExtraeEventTypes.NUMBER_OF_READY_TASKS, self.readyTasksCount))
#
#	def hook_schedulerGetTask(self, event, payload):
#		acquired = event["acquired"]
#		if not acquired:
#			return
#
#		self.readyTasksCount -= 1
#		#print("get: " + str(self.readyTasksCount))
#		#assert(self.readyTasksCount >= 0)
#		payload.append((ExtraeEventTypes.NUMBER_OF_READY_TASKS, self.readyTasksCount))

class ParaverViewCreatedTasks(ParaverView):
	def __init__(self):
		super().__init__()
		self._hooks = [
			("nanos6:task_create_enter", self.hook_taskCreate),
		]
		ParaverTrace.addEventType(ExtraeEventTypes.NUMBER_OF_CREATED_TASKS, "Number of Created Tasks")
		self.createdTasksCount = 0

	def start(self, payload):
		payload.append((ExtraeEventTypes.NUMBER_OF_CREATED_TASKS, 0))

	def hook_taskCreate(self, _, payload):
		self.createdTasksCount += 1
		payload.append((ExtraeEventTypes.NUMBER_OF_CREATED_TASKS, self.createdTasksCount))

class ParaverViewBlockedTasks(ParaverView):
	def __init__(self):
		super().__init__()
		self._hooks = [
			("nanos6:task_block",   self.hook_taskBlock),
			("nanos6:task_execute", self.hook_taskExecute),
		]
		ParaverTrace.addEventType(ExtraeEventTypes.NUMBER_OF_BLOCKED_TASKS, "Number of Blocked Tasks")
		self.blockedTasksCount = 0
		self.blockedTasksIds = set()

	def start(self, payload):
		payload.append((ExtraeEventTypes.NUMBER_OF_BLOCKED_TASKS, 0))

	def hook_taskBlock(self, event, payload):
		taskId = event["id"]
		if taskId in self.blockedTasksIds:
			raise Exception("attempt to block the same task twice")
		self.blockedTasksIds.add(taskId)
		self.blockedTasksCount += 1
		payload.append((ExtraeEventTypes.NUMBER_OF_BLOCKED_TASKS, self.blockedTasksCount))

	def hook_taskExecute(self, event, payload):
		taskId = event["id"]

		try:
			self.blockedTasksIds.remove(taskId)
		except:
			return

		self.blockedTasksCount -= 1
		payload.append((ExtraeEventTypes.NUMBER_OF_BLOCKED_TASKS, self.blockedTasksCount))

class ParaverViewRunningTasks(ParaverView):
	def __init__(self):
		super().__init__()
		self._hooks = [
			("nanos6:task_execute", self.hook_taskExecute),
			("nanos6:task_block",   self.hook_taskStop),
			("nanos6:task_end",     self.hook_taskStop),
		]
		ParaverTrace.addEventType(ExtraeEventTypes.NUMBER_OF_RUNNING_TASKS, "Number of Running Tasks")
		self.runningTasksCount = 0

	def start(self, payload):
		payload.append((ExtraeEventTypes.NUMBER_OF_RUNNING_TASKS, 0))

	def hook_taskExecute(self, _, payload):
		self.runningTasksCount += 1
		payload.append((ExtraeEventTypes.NUMBER_OF_RUNNING_TASKS, self.runningTasksCount))

	def hook_taskStop(self, _, payload):
		self.runningTasksCount -= 1
		payload.append((ExtraeEventTypes.NUMBER_OF_RUNNING_TASKS, self.runningTasksCount))

class ParaverViewCreatedThreads(ParaverView):
	def __init__(self):
		super().__init__()
		self._hooks = [
			("nanos6:thread_create",   self.hook_threadCreate),
			("nanos6:thread_shutdown", self.hook_threadShutdown),
		]
		ParaverTrace.addEventType(ExtraeEventTypes.NUMBER_OF_CREATED_THREADS, "Number of Created Threads")
		self.createdThreadsCount = 0

	def start(self, payload):
		payload.append((ExtraeEventTypes.NUMBER_OF_CREATED_THREADS, 0))

	def hook_threadCreate(self, _, payload):
		self.createdThreadsCount += 1
		payload.append((ExtraeEventTypes.NUMBER_OF_CREATED_THREADS, self.createdThreadsCount))

	def hook_threadShutdown(self, _, payload):
		self.createdThreadsCount -= 1
		assert(self.createdThreadsCount >= 0)
		payload.append((ExtraeEventTypes.NUMBER_OF_CREATED_THREADS, self.createdThreadsCount))

class ParaverViewRunningThreads(ParaverView):
	def __init__(self):
		super().__init__()
		self._hooks = [
			("nanos6:thread_resume",   self.hook_threadResume),
			("nanos6:thread_suspend",  self.hook_threadStop),
			("nanos6:thread_shutdown", self.hook_threadStop),
		]
		ParaverTrace.addEventType(ExtraeEventTypes.NUMBER_OF_RUNNING_THREADS, "Number of Running Threads")
		self.runningThreadsCount = 0

	def start(self, payload):
		payload.append((ExtraeEventTypes.NUMBER_OF_RUNNING_THREADS, 0))

	def hook_threadResume(self, _, payload):
		self.runningThreadsCount += 1
		payload.append((ExtraeEventTypes.NUMBER_OF_RUNNING_THREADS, self.runningThreadsCount))

	def hook_threadStop(self, _, payload):
		self.runningThreadsCount -= 1
		assert(self.runningThreadsCount >= 0)
		payload.append((ExtraeEventTypes.NUMBER_OF_RUNNING_THREADS, self.runningThreadsCount))

class ParaverViewBlockedThreads(ParaverView):
	def __init__(self):
		super().__init__()
		self._hooks = [
			("nanos6:thread_resume",   self.hook_threadResume),
			("nanos6:thread_suspend",  self.hook_threadSuspend),
		]
		ParaverTrace.addEventType(ExtraeEventTypes.NUMBER_OF_BLOCKED_THREADS, "Number of Blocked Threads")
		self.blockedThreads = set()

	def start(self, payload):
		payload.append((ExtraeEventTypes.NUMBER_OF_BLOCKED_THREADS, 0))

	def hook_threadResume(self, event, payload):
		tid = event["tid"]
		try:
			self.blockedThreads.remove(tid)
		except:
			return #it was a new thread, not a previously blocked one
		payload.append((ExtraeEventTypes.NUMBER_OF_BLOCKED_THREADS, len(self.blockedThreads)))

	def hook_threadSuspend(self, event, payload):
		tid = event["tid"]
		if tid in self.blockedThreads:
			raise Exception("Error: attempt to suspend the same thread twice")
		self.blockedThreads.add(tid)
		payload.append((ExtraeEventTypes.NUMBER_OF_BLOCKED_THREADS, len(self.blockedThreads)))

class ParaverViewCTFFlush(ParaverView):
	def __init__(self):
		super().__init__()
		self._hooks = [
			("nanos6:ctf_flush",   self.hook_flush),
		]
		ParaverTrace.addEventTypeAndValue(ExtraeEventTypes.CTF_FLUSH, {1 : "flush"}, "Nanos6 CTF buffers writes to disk")
		self.blockedThreads = set()

	def hook_flush(self, event, _):
		start = event["start"]
		end   = event["end"]

		# In this view we are not emiting events trough the "payload" variable,
		# but we are emitting events directly. That's because we don't want to
		# use the current event timestamp as the extare timestamp but we want
		# to use the event's fields as timestamps. It is safe to do so becaue
		# on flushing, we know that no events could be emitted between the last
		# processed event and now, basically because nanos6 was flushing the
		# buffer :-)

		cpuId = RuntimeModel.getVirtualCPUId(event)
		ParaverTrace.emitEvent(start, cpuId, [(ExtraeEventTypes.CTF_FLUSH, 1)])
		ParaverTrace.emitEvent(end, cpuId, [(ExtraeEventTypes.CTF_FLUSH, 0)])
