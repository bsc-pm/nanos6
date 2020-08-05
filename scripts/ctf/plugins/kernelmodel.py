import json
import re

from executionmodel import ExecutionModel
from paravertrace import ParaverTrace

class CPU():
	def __init__(self, cpuId):
		self._cpuId = cpuId
		self._currentThread = None

	@property
	def currentThread(self):
		return self._currentThread

	@currentThread.setter
	def currentThread(self, value):
		self._currentThread = value

class Thread():
	def __init__(self, tid, name, perProcessExtraeId):
		self._tid = tid
		self._name = name
		self._perProcessExtraeId = perProcessExtraeId
		self.kernelSyscallsEventStack    = [0]
		self.kernelPreemptionsEventStack = [0]

	@property
	def perProcessExtraeId(self):
		return self._perProcessExtraeId

class KernelModel():
	_cpus               = None
	_threads            = {}

	_defs               = []
	_syscalls           = []
	_syscallEntryPoints = []
	_syscallExitPoints  = []

	_enabled            = False
	_preHooks           = []
	_postHooks          = []

	_processNames       = {}
	_processNameId      = None
	_newProcessNameCallbacks = []
	_newThreadCallbacks = []

	@classmethod
	def initialize(cls):
		# Load kernel tracepoint definitions file
		try:
			cls.loadKernelDefs("../nanos6_kerneldefs.json")
		except Exception as e:
			return False

		# Initialize hooks
		cls._preHooks = [
			("sched_switch", cls.hook_schedSwitch)
		]
		cls._postHooks = [
		]

		# Initialize syscall tracepoitns
		regex = re.compile('^sys\_')
		cls._syscalls = list(filter(regex.match, cls._defs))

		regex = re.compile('^sys\_enter\_')
		cls._syscallEntryPoints = list(filter(regex.match, cls._syscalls))

		regex = re.compile('^sys\_exit\_')
		cls._syscallExitPoints = list(filter(regex.match, cls._syscalls))

		# Initialize CPUs
		maxCPUId = ParaverTrace.getMaxRealCPUId()
		cls._cpus = [CPU(cpuId) for cpuId in range(maxCPUId + 1)]

		# Initialize thread list with a fake Idle thread shared by all Cores
		# (all idle threads in the Linux Kernel have tid 0)
		cls._threads[0] = Thread(0, "Idle", 0)

		# Register the traced application's name to ensure that it always gets
		# the same perProcessExtraeId
		binaryName = ParaverTrace.getBinaryName()
		shortBinaryName = binaryName[:15]
		cls._processNames[shortBinaryName] = 1
		cls._processNameId = 100

		cls._enabled = True

		return cls._enabled

	@classmethod
	def getCurrentThread(cls):
		cpuId = ExecutionModel.getCurrentCPUId()
		return cls._cpus[cpuId].currentThread

	@classmethod
	def getCurrentProcessExtraeId(cls):
		thread = cls.getCurrentThread()
		extraeId = thread.perProcessExtraeId
		return extraeId

	@classmethod
	def loadKernelDefs(cls, path):
		with open(path) as f:
			cls._defs = json.load(f)

		if "meta" in cls._defs:
			del cls._defs["meta"]

	@classmethod
	def getSyscallDefinitions(cls):
		return cls._syscallEntryPoints, cls._syscallExitPoints

	@classmethod
	def enabled(cls):
		return cls._enabled

	@classmethod
	def getThread(cls, tid, event):
		""" only callable from the sched_switch hook """
		thread = None
		try:
			thread = cls._threads[tid]
		except:
			# The thread does not exist, let's create it
			perProcessExtraeId = None
			name = event["next_comm"]
			try:
				# Check if the thread command name is already registered.
				# Note that using the process name is not safe to distinguish
				# processes, this is just a process name database, not a
				# process database
				perProcessExtraeId = cls._processNames[name]
			except:
				# The thread command is new, let's assing a new id to it
				perProcessExtraeId     = cls._processNameId
				cls._processNames[name] = cls._processNameId
				cls._processNameId += 1

				for callback in cls._newProcessNameCallbacks:
					callback(name, perProcessExtraeId)
			thread = Thread(tid, name, perProcessExtraeId)
			cls._threads[tid] = thread
			for callback in cls._newThreadCallbacks:
				callback(thread, tid, perProcessExtraeId)
		return thread

	@classmethod
	def registerNewProcessNameCallback(cls, callback):
		cls._newProcessNameCallbacks.append(callback)

	@classmethod
	def registerNewThreadCallback(cls, callback):
		cls._newThreadCallbacks.append(callback)

	@classmethod
	def preHooks(cls):
		return cls._preHooks

	@classmethod
	def postHooks(cls):
		return cls._postHooks

	@classmethod
	def hook_schedSwitch(cls, event, _):
		cpuId = ExecutionModel.getCurrentCPUId()
		tid = event["next_pid"]
		cpu = cls._cpus[cpuId]
		thread = cls.getThread(tid, event)
		cpu.currentThread = thread

