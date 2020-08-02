#
#	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
#
#	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
#

from abc import ABC
from runtime import RuntimeModel
from paravertrace import ParaverTrace, ExtraeEventTypes, ExtraeEventCollection
from hwcdefs import hardwareCountersDefinitions

class RuntimeActivity:
	End         = 0
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
		values = {
			RuntimeActivity.End     : "End",
			RuntimeActivity.Runtime : "Runtime"
		}
		ParaverTrace.addEventTypeAndValue(ExtraeEventTypes.RUNTIME_CODE, values, "Runtime: Runtime Code")

	def hook_threadResume(self, _, payload):
		payload.append((ExtraeEventTypes.RUNTIME_CODE, RuntimeActivity.Runtime))

	def hook_threadStop(self, _, payload):
		payload.append((ExtraeEventTypes.RUNTIME_CODE, RuntimeActivity.End))

class ParaverViewRuntimeBusyWaiting(ParaverView):
	def __init__(self):
		super().__init__()
		self._hooks = [
			("nanos6:thread_shutdown",        self.hook_threadStop),
			("nanos6:thread_suspend",         self.hook_threadStop),
			("nanos6:thread_resume",          self.hook_threadResume),
			("nanos6:worker_enter_busy_wait", self.hook_enterBusyWait),
			("nanos6:worker_exit_busy_wait",  self.hook_exitBusyWait)
		]
		values = {
			RuntimeActivity.End         : "End",
			RuntimeActivity.BusyWaiting : "Busy Waiting",
		}
		ParaverTrace.addEventTypeAndValue(ExtraeEventTypes.RUNTIME_BUSYWAITING, values, "Runtime: Busy Waiting")

	def hook_threadResume(self, event, payload):
		thread = RuntimeModel.getCurrentThread(event)
		if thread.isBusyWaiting:
			payload.append((ExtraeEventTypes.RUNTIME_BUSYWAITING, RuntimeActivity.BusyWaiting))

	def hook_threadStop(self, event, payload):
		thread = RuntimeModel.getCurrentThread(event)
		if thread.isBusyWaiting:
			payload.append((ExtraeEventTypes.RUNTIME_BUSYWAITING, RuntimeActivity.End))

	def hook_enterBusyWait(self, event, payload):
		thread = RuntimeModel.getCurrentThread(event)
		thread.isBusyWaiting = 1
		payload.append((ExtraeEventTypes.RUNTIME_BUSYWAITING, RuntimeActivity.BusyWaiting))

	def hook_exitBusyWait(self, event, payload):
		thread = RuntimeModel.getCurrentThread(event)
		thread.isBusyWaiting = 0
		payload.append((ExtraeEventTypes.RUNTIME_BUSYWAITING, RuntimeActivity.End))

class ParaverViewRuntimeTasks(ParaverView):
	def __init__(self):
		super().__init__()
		self._hooks = [
			("nanos6:task_start",                    self.hook_taskExecute),
			("nanos6:task_end",                      self.hook_taskStop),
			("nanos6:tc:task_create_enter",          self.hook_taskStop),
			("nanos6:tc:task_submit_exit",           self.hook_taskExecute),
			("nanos6:tc:taskwait_enter",             self.hook_taskStop),
			("nanos6:tc:taskwait_exit",              self.hook_taskExecute),
			("nanos6:tc:waitfor_enter",              self.hook_taskStop),
			("nanos6:tc:waitfor_exit",               self.hook_taskExecute),
			("nanos6:tc:mutex_lock_enter",           self.hook_taskStop),
			("nanos6:tc:mutex_lock_exit",            self.hook_taskExecute),
			("nanos6:tc:mutex_unlock_enter",         self.hook_taskStop),
			("nanos6:tc:mutex_unlock_exit",          self.hook_taskExecute),
			("nanos6:tc:blocking_api_block_enter",   self.hook_taskStop),
			("nanos6:tc:blocking_api_block_exit",    self.hook_taskExecute),
			("nanos6:tc:blocking_api_unblock_enter", self.hook_taskStop),
			("nanos6:tc:blocking_api_unblock_exit",  self.hook_taskExecute),
			("nanos6:tc:spawn_function_enter",       self.hook_taskStop),
			("nanos6:tc:spawn_function_exit",        self.hook_taskExecute),
		]
		values = {
			RuntimeActivity.End  : "End",
			RuntimeActivity.Task : "Task",
		}
		ParaverTrace.addEventTypeAndValue(ExtraeEventTypes.RUNTIME_TASKS, values, "Runtime: Task Code")

	def hook_taskExecute(self, _, payload):
		payload.append((ExtraeEventTypes.RUNTIME_TASKS, RuntimeActivity.Task))

	def hook_taskStop(self, _, payload):
		payload.append((ExtraeEventTypes.RUNTIME_TASKS, RuntimeActivity.End))

class ParaverViewTaskLabel(ParaverView):
	""" Shows the label of tasks running on each core. Periods of runtime code
	running on behalf of tasks are shown as tasks unless the task blocks or
	ends. A worker might suspend while running a task that has not blocked,
	this happens when the task does not block, but the runtime considers that
	another worker must run instead of the current one. In this case, the task
	is added into the ready queue and we do not see a block event. We detect
	this case by capturing the thread_suspend and thread_resume tracepoints"""

	def __init__(self):
		super().__init__()
		self._hooks = [
			("nanos6:task_label",     self.hook_taskLabel),
			("nanos6:task_start",     self.hook_taskExecute),
			("nanos6:task_block",     self.hook_taskStop),
			("nanos6:task_unblock",   self.hook_taskExecute),
			("nanos6:task_end",       self.hook_taskStop),
			("nanos6:thread_resume",  self.hook_threadResume),
			("nanos6:thread_suspend", self.hook_threadSuspend),
		]
		values = {
			RuntimeActivity.End : "End",
		}
		ParaverTrace.addEventTypeAndValue(ExtraeEventTypes.RUNNING_TASK_LABEL, values, "Running Task Label")

	def hook_taskLabel(self, event, _):
		label      = event["label"]
		taskTypeID = event["type"]
		ParaverTrace.addEventTypeAndValue(ExtraeEventTypes.RUNNING_TASK_LABEL, {taskTypeID : label})

	def hook_taskExecute(self, event, payload):
		task = RuntimeModel.getCurrentTask(event)
		payload.append((ExtraeEventTypes.RUNNING_TASK_LABEL, task.type))

	def hook_taskStop(self, event, payload):
		payload.append((ExtraeEventTypes.RUNNING_TASK_LABEL, RuntimeActivity.End))

	def hook_threadResume(self, event, payload):
		task = RuntimeModel.getCurrentTask(event)
		if task.isRunning():
			payload.append((ExtraeEventTypes.RUNNING_TASK_LABEL, task.type))

	def hook_threadSuspend(self, event, payload):
		task = RuntimeModel.getCurrentTask(event)
		if task.isRunning():
			payload.append((ExtraeEventTypes.RUNNING_TASK_LABEL, RuntimeActivity.End))

class ParaverViewTaskSource(ParaverView):
	""" Shows the source code location of running tasks on each core. See
	ParaverViewTaskLabel for more details. """

	def __init__(self):
		super().__init__()
		self._hooks = [
			("nanos6:task_label",     self.hook_taskLabel),
			("nanos6:task_start",     self.hook_taskExecute),
			("nanos6:task_end",       self.hook_taskStop),
			("nanos6:task_block",     self.hook_taskStop),
			("nanos6:task_unblock",   self.hook_taskExecute),
			("nanos6:thread_resume",  self.hook_threadResume),
			("nanos6:thread_suspend", self.hook_threadSuspend),
		]
		values = {
			RuntimeActivity.End : "End",
		}
		ParaverTrace.addEventTypeAndValue(ExtraeEventTypes.RUNNING_TASK_SOURCE, values, "Running Task Source")

	def hook_taskLabel(self, event, _):
		source     = event["source"]
		taskTypeID = event["type"]
		ParaverTrace.addEventTypeAndValue(ExtraeEventTypes.RUNNING_TASK_SOURCE, {taskTypeID : source})

	def hook_taskExecute(self, event, payload):
		task = RuntimeModel.getCurrentTask(event)
		payload.append((ExtraeEventTypes.RUNNING_TASK_SOURCE, task.type))

	def hook_taskStop(self, event, payload):
		payload.append((ExtraeEventTypes.RUNNING_TASK_SOURCE, RuntimeActivity.End))

	def hook_threadResume(self, event, payload):
		task = RuntimeModel.getCurrentTask(event)
		if task.isRunning():
			payload.append((ExtraeEventTypes.RUNNING_TASK_SOURCE, task.type))

	def hook_threadSuspend(self, event, payload):
		task = RuntimeModel.getCurrentTask(event)
		if task.isRunning():
			payload.append((ExtraeEventTypes.RUNNING_TASK_SOURCE, RuntimeActivity.End))

class ParaverViewTaskId(ParaverView):
	""" Shows the task Id of running tasks on each core. See
	ParaverViewTaskLabel for more details. """

	def __init__(self):
		super().__init__()
		self._hooks = [
			("nanos6:task_start",     self.hook_taskExecute),
			("nanos6:task_end",       self.hook_taskStop),
			("nanos6:task_block",     self.hook_taskStop),
			("nanos6:task_unblock",   self.hook_taskExecute),
			("nanos6:thread_resume",  self.hook_threadResume),
			("nanos6:thread_suspend", self.hook_threadSuspend),
		]
		values = {
			RuntimeActivity.End : "End",
		}
		ParaverTrace.addEventTypeAndValue(ExtraeEventTypes.RUNNING_TASK_ID, values, "Task ID")

	def hook_taskExecute(self, event, payload):
		task = RuntimeModel.getCurrentTask(event)
		payload.append((ExtraeEventTypes.RUNNING_TASK_ID, task.id))

	def hook_taskStop(self, event, payload):
		payload.append((ExtraeEventTypes.RUNNING_TASK_ID, RuntimeActivity.End))

	def hook_threadResume(self, event, payload):
		task = RuntimeModel.getCurrentTask(event)
		if task.isRunning():
			payload.append((ExtraeEventTypes.RUNNING_TASK_ID, task.id))

	def hook_threadSuspend(self, event, payload):
		task = RuntimeModel.getCurrentTask(event)
		if task.isRunning():
			payload.append((ExtraeEventTypes.RUNNING_TASK_ID, RuntimeActivity.End))

class ParaverViewHardwareCounters(ParaverView):
	def __init__(self):
		super().__init__()
		self._hooks = [
			("nanos6:thread_suspend",                self.hook_getHardwareCounters),
			("nanos6:thread_shutdown",               self.hook_getHardwareCounters),
			("nanos6:task_start",                    self.hook_getHardwareCounters),
			("nanos6:task_end",                      self.hook_getHardwareCounters),
			("nanos6:tc:task_create_enter",          self.hook_getHardwareCounters),
			("nanos6:tc:task_submit_exit",           self.hook_getHardwareCounters),
			("nanos6:tc:taskwait_enter",             self.hook_getHardwareCounters),
			("nanos6:tc:taskwait_exit",              self.hook_getHardwareCounters),
			("nanos6:tc:waitfor_enter",              self.hook_getHardwareCounters),
			("nanos6:tc:waitfor_exit",               self.hook_getHardwareCounters),
			("nanos6:tc:mutex_lock_enter",           self.hook_getHardwareCounters),
			("nanos6:tc:mutex_lock_exit",            self.hook_getHardwareCounters),
			("nanos6:tc:mutex_unlock_enter",         self.hook_getHardwareCounters),
			("nanos6:tc:mutex_unlock_exit",          self.hook_getHardwareCounters),
			("nanos6:tc:blocking_api_block_enter",   self.hook_getHardwareCounters),
			("nanos6:tc:blocking_api_block_exit",    self.hook_getHardwareCounters),
			("nanos6:tc:blocking_api_unblock_enter", self.hook_getHardwareCounters),
			("nanos6:tc:blocking_api_unblock_exit",  self.hook_getHardwareCounters),
			("nanos6:tc:spawn_function_enter",       self.hook_getHardwareCounters),
			("nanos6:tc:spawn_function_exit",        self.hook_getHardwareCounters),
		]

		self._eventsHardwareCounters = ExtraeEventCollection(3900000, 7)
		self._eventsHardwareCounters.addEvents(hardwareCountersDefinitions)
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
		values = {
			RuntimeActivity.End : "End",
		}
		ParaverTrace.addEventTypeAndValue(ExtraeEventTypes.RUNNING_THREAD_TID, values, "Worker Thread Id (TID)")

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
		payload.append((ExtraeEventTypes.RUNNING_THREAD_TID, RuntimeActivity.End))

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
		TaskWait             = 12
		WaitFor              = 13
		Lock                 = 14
		Unlock               = 15
		BlockingAPIBlock     = 16
		BlockingAPIUnblock   = 17
		SpawnFunction        = 18
		SchedulerLockEnter   = 19
		SchedulerLockServing = 20
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
			("nanos6:thread_create",                           self.hook_initStack),
			("nanos6:external_thread_create",                  self.hook_initExternalThreadStack),
			("nanos6:thread_resume",                           self.hook_eventContinue),
			("nanos6:external_thread_resume",                  self.hook_runtime),
			("nanos6:thread_suspend",                          self.hook_eventStop),
			("nanos6:external_thread_suspend",                 self.hook_unstack),
			("nanos6:thread_shutdown",                         self.hook_eventStop),
			("nanos6:external_thread_shutdown",                self.hook_eventStop),
			("nanos6:task_start",                              self.hook_task),
			("nanos6:task_end",                                self.hook_unstack),
			("nanos6:tc:taskwait_enter",                       self.hook_taskWait),
			("nanos6:tc:taskwait_exit",                        self.hook_unstack),
			("nanos6:tc:waitfor_enter",                        self.hook_waitFor),
			("nanos6:tc:waitfor_exit",                         self.hook_unstack),
			("nanos6:tc:mutex_lock_enter",                     self.hook_lock),
			("nanos6:tc:mutex_lock_exit",                      self.hook_unstack),
			("nanos6:tc:mutex_unlock_enter",                   self.hook_unlock),
			("nanos6:tc:mutex_unlock_exit",                    self.hook_unstack),
			("nanos6:tc:blocking_api_block_enter",             self.hook_blockingAPIBlock),
			("nanos6:tc:blocking_api_block_exit",              self.hook_unstack),
			("nanos6:tc:blocking_api_unblock_enter",           self.hook_blockingAPIUnblock),
			("nanos6:tc:blocking_api_unblock_exit",            self.hook_unstack),
			("nanos6:oc:blocking_api_unblock_enter",           self.hook_blockingAPIUnblock),
			("nanos6:oc:blocking_api_unblock_exit",            self.hook_unstack),
			("nanos6:tc:spawn_function_enter",                 self.hook_spawnFunction),
			("nanos6:tc:spawn_function_exit",                  self.hook_unstack),
			("nanos6:oc:spawn_function_enter",                 self.hook_spawnFunction),
			("nanos6:oc:spawn_function_exit",                  self.hook_unstack),
			("nanos6:worker_enter_busy_wait",                  self.hook_busyWait),
			("nanos6:worker_exit_busy_wait",                   self.hook_unstack),
			("nanos6:dependency_register_enter",               self.hook_dependencyRegister),
			("nanos6:dependency_register_exit",                self.hook_unstack),
			("nanos6:dependency_unregister_enter",             self.hook_dependencyUnregister),
			("nanos6:dependency_unregister_exit",              self.hook_unstack),
			("nanos6:scheduler_add_task_enter",                self.hook_schedulerAddTask),
			("nanos6:scheduler_add_task_exit",                 self.hook_unstack),
			("nanos6:scheduler_get_task_enter",                self.hook_schedulerGetTask),
			("nanos6:scheduler_get_task_exit",                 self.hook_unstack),
			("nanos6:tc:task_create_enter",                    self.hook_taskCreate),
			("nanos6:tc:task_create_exit",                     self.hook_taskBetweenCreateAndSubmit),
			("nanos6:oc:task_create_enter",                    self.hook_taskCreate),
			("nanos6:oc:task_create_exit",                     self.hook_taskBetweenCreateAndSubmit),
			("nanos6:tc:task_submit_enter",                    self.hook_taskSubmit),
			("nanos6:tc:task_submit_exit",                     self.hook_unstack),
			("nanos6:oc:task_submit_enter",                    self.hook_taskSubmit),
			("nanos6:oc:task_submit_exit",                     self.hook_unstack),
			("nanos6:taskfor_init_enter",                      self.hook_taskforInit),
			("nanos6:taskfor_init_exit",                       self.hook_unstack),
			("nanos6:scheduler_lock_client",                   self.hook_schedLockClient),
			("nanos6:scheduler_lock_server",                   self.hook_schedLockServer),
			("nanos6:scheduler_lock_assign",                   self.hook_schedLockAssign),
			("nanos6:scheduler_lock_server_exit",              self.hook_unstack),
			("nanos6:debug_register",                          self.hook_debugRegister),
			("nanos6:debug_enter",                             self.hook_debug),
			("nanos6:debug_exit",                              self.hook_unstack),
		]
		status = {
			self.Status.Idle:                             "Idle",
			self.Status.Runtime:                          "Runtime",
			self.Status.BusyWait:                         "Busy Wait",
			self.Status.Task:                             "Task",
			self.Status.DependencyRegister:               "Dependency: Register",
			self.Status.DependencyUnregister:             "Dependency: Unregister",
			self.Status.SchedulerAddTask:                 "Scheduler: Add Ready Task",
			self.Status.SchedulerGetTask:                 "Scheduler: Get Ready Task",
			self.Status.TaskCreate:                       "Task: Create",
			self.Status.TaskArgsInit:                     "Task: Arguments Init",
			self.Status.TaskSubmit:                       "Task: Submit",
			self.Status.TaskforInit:                      "Task: Taskfor Collaborator Init",
			self.Status.TaskWait:                         "Task: TaskWait",
			self.Status.WaitFor:                          "Task: WaitFor",
			self.Status.Lock:                             "Task: User Mutex: Lock",
			self.Status.Unlock:                           "Task: User Mutex: Unlock",
			self.Status.BlockingAPIBlock:                 "Task: Blocking API: Block",
			self.Status.BlockingAPIUnblock:               "Task: Blocking API: Unblock",
			self.Status.SpawnFunction:                    "SpawnFunction: Spawn",
			self.Status.SchedulerLockEnter:               "Scheduler: Lock: Enter",
			self.Status.SchedulerLockServing:             "Scheduler: Lock: Serving tasks",
		}
		ParaverTrace.addEventTypeAndValue(ExtraeEventTypes.RUNTIME_SUBSYSTEMS, status, "Runtime Subsystems")
		self._servedTasks = {}

	def hook_initStack(self, event, payload):
		tid = event["tid"]
		thread = RuntimeModel.getThread(tid)
		thread.eventStack = [self.Status.Runtime]

	def hook_initExternalThreadStack(self, event, payload):
		tid = event["tid"]
		thread = RuntimeModel.getThread(tid)
		thread.eventStack = [self.Status.Idle]

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
	def hook_runtime(self, _):
		return self.Status.Runtime

	@stackEvent
	def hook_task(self, _):
		return self.Status.Task

	@stackEvent
	def hook_taskWait(self, _):
		return self.Status.TaskWait

	@stackEvent
	def hook_waitFor(self, _):
		return self.Status.WaitFor

	@stackEvent
	def hook_lock(self, _):
		return self.Status.Lock

	@stackEvent
	def hook_unlock(self, _):
		return self.Status.Unlock

	@stackEvent
	def hook_blockingAPIBlock(self, _):
		return self.Status.BlockingAPIBlock

	@stackEvent
	def hook_blockingAPIUnblock(self, _):
		return self.Status.BlockingAPIUnblock

	@stackEvent
	def hook_spawnFunction(self, _):
		return self.Status.SpawnFunction

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

	def hook_schedLockClient(self, event, payload):
		taskId  = event["id"]
		timeAcq = event["ts_acquire"]
		cpuId   = event["cpu_id"]
		stack   = self.getEventStack(event)

		# Emit event in the past. We know that no tracepoint can be emitted
		# between getting the lock and after the operation completes
		# (at this point as a client). Hence, it is safe the override the
		# standard path and enforce another tracepoint here.
		pastPayload = [(ExtraeEventTypes.RUNTIME_SUBSYSTEMS, self.Status.SchedulerLockEnter)]
		ParaverTrace.emitEvent(timeAcq, cpuId, pastPayload)

		# Emit communication line (if present) before any current event as
		# requried by paraver. This line shows a relation between a worker
		# serving task and another worker being assigned work by it.
		if taskId != 0:
			timeRel = RuntimeModel.getCurrentTimestamp()
			(timeSend, cpuSendId) = self._servedTasks[taskId]
			ParaverTrace.emitCommunicationEvent(cpuSendId, timeSend, cpuId, timeRel)

		# Emit event at this point in time using the standard path
		payload.append((ExtraeEventTypes.RUNTIME_SUBSYSTEMS, stack[-1]))

	def hook_schedLockServer(self, event, payload):
		timeAcq = event["ts_acquire"]
		cpuId   = event["cpu_id"]
		stack   = self.getEventStack(event)

		# Emit event in the past. We know that no tracepoint can be emitted
		# between before getting the lock and after the operation completes
		# (at this point as a server). Hence, it is safe the override the
		# standard path and enforce another tracepoint here.
		pastPayload = [(ExtraeEventTypes.RUNTIME_SUBSYSTEMS, self.Status.SchedulerLockEnter)]
		ParaverTrace.emitEvent(timeAcq, cpuId, pastPayload)

		# Emit event at this point in time
		payload.append((ExtraeEventTypes.RUNTIME_SUBSYSTEMS, self.Status.SchedulerLockServing))

		# Add current new status to the event stack
		stack.append(self.Status.SchedulerLockServing)

	def hook_schedLockAssign(self, event, _):
		taskId = event["id"]
		cpuSendId = event["cpu_id"]
		self._servedTasks[taskId] = (RuntimeModel.getCurrentTimestamp(), cpuSendId)

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

class ParaverViewCTFFlush(ParaverView):
	def __init__(self):
		super().__init__()
		self._hooks = [
			("nanos6:ctf_flush",   self.hook_flush),
		]
		values = {
			0 : "End",
			1 : "flush",
		}
		ParaverTrace.addEventTypeAndValue(ExtraeEventTypes.CTF_FLUSH, values, "Nanos6 CTF buffers writes to disk")
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

#class ParaverViewNumberOfReadyTasks(ParaverView):
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

class ParaverViewNumberOfCreatedTasks(ParaverView):
	def __init__(self):
		super().__init__()
		self._hooks = [
			("nanos6:tc:task_create_enter", self.hook_taskCreate),
			("nanos6:oc:task_create_enter", self.hook_taskCreate),
		]
		ParaverTrace.addEventType(ExtraeEventTypes.NUMBER_OF_CREATED_TASKS, "Number of Created Tasks")
		self.createdTasksCount = 0

	def start(self, payload):
		payload.append((ExtraeEventTypes.NUMBER_OF_CREATED_TASKS, 0))

	def hook_taskCreate(self, _, payload):
		self.createdTasksCount += 1
		payload.append((ExtraeEventTypes.NUMBER_OF_CREATED_TASKS, self.createdTasksCount))

class ParaverViewNumberOfBlockedTasks(ParaverView):
	def __init__(self):
		super().__init__()
		self._hooks = [
			("nanos6:task_block",   self.hook_taskBlock),
			("nanos6:task_unblock", self.hook_taskUnblock),
		]
		ParaverTrace.addEventType(ExtraeEventTypes.NUMBER_OF_BLOCKED_TASKS, "Number of Blocked Tasks")
		self.blockedTasksCount = 0

	def start(self, payload):
		payload.append((ExtraeEventTypes.NUMBER_OF_BLOCKED_TASKS, 0))

	def hook_taskBlock(self, event, payload):
		self.blockedTasksCount += 1
		payload.append((ExtraeEventTypes.NUMBER_OF_BLOCKED_TASKS, self.blockedTasksCount))

	def hook_taskUnblock(self, event, payload):
		self.blockedTasksCount -= 1
		payload.append((ExtraeEventTypes.NUMBER_OF_BLOCKED_TASKS, self.blockedTasksCount))

class ParaverViewNumberOfRunningTasks(ParaverView):
	def __init__(self):
		super().__init__()
		self._hooks = [
			("nanos6:task_start",   self.hook_taskExecute),
			("nanos6:task_block",   self.hook_taskStop),
			("nanos6:task_unblock", self.hook_taskExecute),
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

class ParaverViewNumberOfCreatedThreads(ParaverView):
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

class ParaverViewNumberOfRunningThreads(ParaverView):
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

class ParaverViewNumberOfBlockedThreads(ParaverView):
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

class ParaverViewKernelThreadID(ParaverView):
	def __init__(self):
		super().__init__()
		self._hooks = [
			("sched_switch", self.hook_schedSwitch),
		]
		ParaverTrace.addEventType(ExtraeEventTypes.KERNEL_THREAD_ID, "Kernel Thread ID (TID)")

	def hook_schedSwitch(self, event, payload):
		tid = event["next_pid"]
		# Idle threads have Id 0, which correspond to the "End" state in
		# Paraver. Therfore, no special treament need to be done for the idle
		# threads.
		payload.append((ExtraeEventTypes.KERNEL_THREAD_ID, tid))

class ParaverViewKernelProcessName(ParaverView):

	class ProcessNameDB():

		def __init__(self, keys):
			self._dic = {}
			self._cnt = 1
			for k in keys:
				val = self._cnt
				self._dic[k] = val
				self._cnt += 1
				ParaverTrace.addEventTypeAndValue(ExtraeEventTypes.KERNEL_PROCESS_NAME, {val : k})

		def __getitem__(self, key):
			val = None
			if key in self._dic:
				val = self._dic[key]
			elif key[:8] == 'swapper/':
				# To detect Idle threads we could also check for tid == 0, but
				# It's probably less expensive to check the string rather than
				# access the event's dictionary
				self._dic[key] = 0
				val = 0
			else:
				val = self._cnt
				self._dic[key] = val
				self._cnt += 1
				ParaverTrace.addEventTypeAndValue(ExtraeEventTypes.KERNEL_PROCESS_NAME, {val : key})
			return val

	def __init__(self):
		super().__init__()
		self._hooks = [
			("sched_switch", self.hook_schedSwitch),
		]

		# Only the idle thread need to be registered manually, all the others
		# processes will be registered in ProcessNameDB
		values = {
			0  : "Idle"
		}
		ParaverTrace.addEventTypeAndValue(ExtraeEventTypes.KERNEL_PROCESS_NAME, values, "Kernel Process Name")

		# Initialize process name to id mapping
		# Add first the traced application's name to ensure that it always gets the same id
		binaryName = ParaverTrace.getBinaryName()
		shortBinaryName = binaryName[:15]
		self._processNameMap = self.ProcessNameDB([shortBinaryName])

	def hook_schedSwitch(self, event, payload):
		name = event["next_comm"]
		extraeId = self._processNameMap[name]
		payload.append((ExtraeEventTypes.KERNEL_PROCESS_NAME, extraeId))

