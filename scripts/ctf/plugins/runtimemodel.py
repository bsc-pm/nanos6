#
#	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
#
#	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
#

from executionmodel import ExecutionModel
from paravertrace   import ParaverTrace
from enum import Enum

class WorkerType(Enum):
	WorkerThread   = 1
	LeaderThread   = 2
	ExternalThread = 3

class CPU:
	id_index = 0
	def __init__(self):
		self._id = CPU.id_index
		CPU.id_index += 1
		self._currentThread = None

	@property
	def currentThread(self):
		return self._currentThread

	@currentThread.setter
	def currentThread(self, value):
		self._currentThread = value

	@property
	def id(self):
		return self._id

	@property
	def isVirtual(self):
		return False

class VCPU(CPU):
	def __init__(self):
		super().__init__()

	@property
	def externalThread(self):
		return self._externalThread

	@externalThread.setter
	def externalThread(self, value):
		self._externalThread = value

	@property
	def isVirtual(self):
		return True

class Task:
	class Status:
		Uninitialized = 0
		Running       = 1
		Blocked       = 2

	def __init__(self):
		self.reset()

	@property
	def id(self):
		return self._id

	@property
	def type(self):
		return self._type

	def isUnitialized(self):
		return self._status == self.Status.Uninitialized

	def reset(self):
		self._id   = -1
		self._type = -1
		self._status = self.Status.Uninitialized

	def start(self, taskId, taskTypeId):
		assert(self._status == self.Status.Uninitialized)
		self._id     = taskId
		self._type   = taskTypeId
		self._status = self.Status.Running

	def block(self):
		assert(self._status == self.Status.Running)
		self._status = self.Status.Blocked

	def unblock(self):
		assert(self._status == self.Status.Blocked)
		self._status = self.Status.Running

	def end(self):
		assert(self._status == self.Status.Running)
		self.reset()

	def isRunning(self):
		return self._status == self.Status.Running

class Thread:
	def __init__(self, tid = 0):
		self._id            = tid
		self._vcpu          = None
		self._isBusyWaiting = 0
		self._tasks         = [Task()]

	@property
	def tid(self):
		return self._id

	@property
	def vcpu(self):
		return self._vcpu

	@vcpu.setter
	def vcpu(self, value):
		self._vcpu = value

	@property
	def task(self):
		return self._tasks[-1]

	@property
	def isBusyWaiting(self):
		return self._isBusyWaiting

	@isBusyWaiting.setter
	def isBusyWaiting(self, value):
		self._isBusyWaiting = value

	def startTask(self, taskId, taskTypeId):
		if not self._tasks[-1].isUnitialized():
			# There was a task assigned to this thread already, hence we must
			# be trying to execute an if0 task while holding another task
			self._tasks.append(Task())
		self._tasks[-1].start(taskId, taskTypeId)

	def endTask(self):
		if len(self._tasks) != 1:
			# if the list of tasks is greater than one, it means we were running
			# an if0 task. In that case, just pop the current task.
			self._tasks.pop()
		else:
			# otherwise, reuse the task object by reinitializing it
			self._tasks[-1].end()

class TaskIDsDB:
	def __init__(self):
		self.__db = {}

	def addEntry(self, taskTypeID, taskID):
		if (taskID in self.__db):
			raise Exception("Attempt to add another different label for task ID: " + str(taskTypeID) + " old label = " + self.__db[taskTypeID] + " new label = " + label)                                                                      
		self.__db[taskID] = taskTypeID

	def getEntry(self, taskID):
		try:
			taskTypeID = self.__db[taskID]
		except:
			print("Error: cannot find taskID to taskTypeID translation into database for task ID " + str(taskID))
			raise
		return taskTypeID

class RuntimeModel:
	_threads = {}
	_taskTypes = TaskIDsDB()

	@classmethod
	def initialize(cls):
		cls._maxCPUId = ParaverTrace.getMaxRealCPUId()
		cls._ltcpu = ParaverTrace.getLeaderThreadCPUId()
		cls._cpus = [CPU() for i in range(cls._maxCPUId + 1)]
		cls._cpus.append(VCPU()) # Leader Thread CPU
		cls._preHooks = [
			("nanos6:tc:task_create_enter",   cls.hook_taskAdd),
			("nanos6:oc:task_create_enter",   cls.hook_taskAdd),
			("nanos6:taskfor_init_enter",     cls.hook_taskAdd),
			("nanos6:external_thread_create", cls.hook_externalThreadCreate),
			("nanos6:thread_create",          cls.hook_threadCreate),
			("nanos6:thread_resume",          cls.hook_threadResume),
			("nanos6:thread_resume",          cls.hook_threadResume),
			("nanos6:task_start",             cls.hook_taskStart),
			("nanos6:task_unblock",           cls.hook_taskUnblock),
		]
		cls._postHooks = [
			("nanos6:thread_suspend",         cls.hook_threadSuspend),
			("nanos6:task_block",             cls.hook_taskBlock),
			("nanos6:task_end",               cls.hook_taskEnd),
		]

	@classmethod
	def getTaskTypeId(cls, taskID):
		return cls._taskTypes.getEntry(taskID)

	@classmethod
	def getWorkerType(cls, vcpuid):
		threadType = None
		if vcpuid <= cls._maxCPUId:
			threadType = WorkerType.WorkerThread
		elif vcpuid == cls._ltcpu:
			threadType = WorkerType.LeaderThread
		else:
			threadType = WorkerType.ExternalThread
		return threadType

	@classmethod
	def getVirtualCPU(cls, event):
		# bounded threads (worker threads) cpu_id event value always points to
		# the physical cpu id. All External Threads have the same cpu_id (all
		# of them share a stream). To distinguish them, we need to check their
		# tid event value. With its tid, we keep track of which "virtual cpu"
		# each of them belong. The Leader Thread has its own stream, and hence
		# it's cpu_id is already a valid virtual cpu_id (=maxCPUId+1)

		vcpuid = ExecutionModel.getCurrentCPUId()
		wtype = cls.getWorkerType(vcpuid)
		if wtype == WorkerType.WorkerThread:
			cp = cls._cpus[vcpuid]
		elif wtype == WorkerType.LeaderThread:
			cp = cls._cpus[vcpuid]
		else: # WorkerType.ExternalThread
			tid = event["unbounded"]["tid"]
			thread = cls.getThread(tid)
			cp = thread.vcpu
			if cp == None:
				cp = VCPU()
				cp.externalThread = thread
				thread.vcpu = cp
				cls._cpus.append(cp)
				ParaverTrace.increaseVirtualCPUCount()
		return cp

	@classmethod
	def getCurrentThread(cls, event):
		vcpu = cls.getVirtualCPU(event)
		thread = None
		if not vcpu.isVirtual:
			thread = vcpu.currentThread
		else:
			thread = vcpu.externalThread
		assert(thread != None)
		return thread

	@classmethod
	def getCurrentTask(cls, event):
		thread = cls.getCurrentThread(event)
		task = thread.task
		return task

	@classmethod
	def getVirtualCPUId(cls, event):
		vcpu = cls.getVirtualCPU(event)
		return vcpu.id

	@classmethod
	def getThread(cls, tid):
		if not tid in cls._threads.keys():
			thread = Thread(tid)
			cls._threads[tid] = thread
		else:
			thread = cls._threads[tid]
		return thread

	@classmethod
	def preHooks(cls):
		return cls._preHooks

	@classmethod
	def postHooks(cls):
		return cls._postHooks

	@classmethod
	def hook_externalThreadCreate(cls, event, _):
		vcpuid = ExecutionModel.getCurrentCPUId()
		wtype = cls.getWorkerType(vcpuid)
		if wtype == WorkerType.LeaderThread:
			tid = event["tid"]
			thread = cls.getThread(tid)
			vcpu = cls._cpus[vcpuid]
			vcpu.externalThread = thread
			thread.vcpu = vcpu

	@classmethod
	def hook_threadCreate(cls, event, _):
		tid = event["tid"]
		thread = cls.getThread(tid)
		assert(thread != None)

	@classmethod
	def hook_taskAdd(cls, event, _):
		taskId     = event["id"]
		taskTypeId = event["type"]
		cls._taskTypes.addEntry(taskTypeId, taskId)

	@classmethod
	def hook_threadResume(cls, event, _):
		tid = event["tid"]
		thread = cls.getThread(tid)
		cpu = cls.getVirtualCPU(event)
		cpu.currentThread = thread

	@classmethod
	def hook_threadSuspend(cls, event, _):
		tid = event["tid"]

		# a thread that suspends must have been created.
		assert(tid in cls._threads.keys())
		# and our thread must be attached to the current cpu
		vcpu = cls.getVirtualCPU(event)
		thread = vcpu.currentThread
		assert(thread != None)
		assert(thread.tid == tid)
		vcpu.currentThread = None

	@classmethod
	def hook_taskStart(cls, event, _):
		taskId     = event["id"]
		taskTypeId = cls.getTaskTypeId(taskId)
		thread     = cls.getCurrentThread(event)
		thread.startTask(taskId, taskTypeId)

	@classmethod
	def hook_taskBlock(cls, event, _):
		thread = cls.getCurrentThread(event)
		thread.task.block()

	@classmethod
	def hook_taskUnblock(cls, event, _):
		thread = cls.getCurrentThread(event)
		thread.task.unblock()

	@classmethod
	def hook_taskEnd(cls, event, _):
		thread = cls.getCurrentThread(event)
		thread.endTask()

