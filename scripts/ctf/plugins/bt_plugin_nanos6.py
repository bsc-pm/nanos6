#
#	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
#
#	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
#

import bt2
import atexit
import os
import sys
import readline
import code
from pprint import pprint
from collections import defaultdict
import operator
import signal

from executionmodel import ExecutionModel
from runtimemodel   import RuntimeModel
from kernelmodel    import KernelModel
from paravertrace import ParaverTrace, ExtraeEventTypes
import paraverviews as pv

bt2.register_plugin(
	module_name=__name__,
	name="nanos6",
	description="Nanos6 CTF instrumentation backend utilities",
	author="Aleix Roca Nonell",
	license="GPL",
	version=(1, 0, 0),
)


class ExitHandler():
	def __init__(self):
		self._exit = False

		# Register exit signals
		signal.signal(signal.SIGTERM, self._sigtermHandler)
		signal.signal(signal.SIGINT,  self._sigintHandler)

		# Register timeout if requested
		timeout = os.environ.get('CTF2PRV_TIMEOUT', "0")
		if timeout != "0" and timeout != 0:
			seconds = 0
			try:
				seconds = int(timeout) * 60
			except:
				raise RuntimeError("Cannot convert CTF2PRV_TIMEOUT value to seconds")
			signal.signal(signal.SIGALRM, self._sigalrmHandler)
			signal.alarm(seconds)

	def exit(self):
		return self._exit

	def _sigalrmHandler(self, signum, frame):
		self._exit = True

	def _sigtermHandler(self, signum, frame):
		self._exit = True

	def _sigintHandler(self, signum, frame):
		self._exit = True

@bt2.plugin_component_class
class ctf2prv(bt2._UserSinkComponent):
	def __init__(self, config, params, obj):
		self.__port = self._add_input_port("in")
		self.__process_message = self._process_first_message
		self.__last = None
		atexit.register(self._finalize)
		self.__payload = []
		self.__verbose = False
		self.__paraverViews = None
		self.__exitHandler = ExitHandler()

		self.__hooks = defaultdict(list)

		verbose = os.environ.get('CTF2PRV_VERBOSE', "0")
		if verbose == "0" or verbose == 0:
			self.__verbose = False
			print("Starting CTF to PRV conversion. Set environemnt variable CTF2PRV_VERBOSE=1 to show progress", flush=True)
		elif verbose == "1" or verbose == 1:
			self.__verbose = True
			print("Starting CTF to PRV conversion", flush=True)
		else:
			raise RuntimeError("Error: Unknown CTF2PRV_VERBOSE value. Expected 0 or 1")

	def _finalize(self):
		ts = self.__last.default_clock_snapshot.value
		ParaverTrace.addEndTime(ts)
		ParaverTrace.finalizeTraceFiles()

	def _user_graph_is_configured(self):
		self._it = self._create_message_iterator(self.__port)

	def _process_event(self, ts, cpuId, event):
		ExecutionModel.setCurrentEventData(ts, cpuId)
		hookList = self.__hooks[event.name]
		for hook in hookList:
			hook(event, self.__payload)
		if self.__payload:
			cpuId = RuntimeModel.getVirtualCPUId(event)
			ParaverTrace.emitEvent(ts, cpuId, self.__payload)
		self.__payload.clear()

	def _user_consume(self):
		self.__process_message()

	def _process_first_message(self):
		msg = next(self._it)
		if type(msg) is bt2._EventMessageConst:
			clk = msg.default_clock_snapshot
			absoluteStartTime = clk.clock_class.offset.seconds
			ts = clk.value - 1 # see comments below
			assert(ts >= 0)
			ncpus = msg.event.stream.trace.environment["ncpus"]
			binaryName = msg.event.stream.trace.environment["binary_name"]
			pid = msg.event.stream.trace.environment["pid"]
			traceName = "trace_" + str(binaryName) + "_" + str(pid)

			# initialize paraver trace
			ParaverTrace.addTraceName(traceName)
			ParaverTrace.addAbsoluteStartTime(absoluteStartTime)
			ParaverTrace.addStartTime(ts)
			ParaverTrace.addNumberOfCPUs(ncpus)
			ParaverTrace.addBinaryName(binaryName)
			ParaverTrace.initalizeTraceFiles()

			# Initialize both Kernel and Runtime Models. The Kernel Model must
			# be initalized before Paravare views are created
			KernelModel.initialize(ncpus)
			RuntimeModel.initialize(ncpus)

			# Create Paraver Views
			self.__paraverViews = [
				pv.ParaverViewRuntimeCode(),
				pv.ParaverViewRuntimeBusyWaiting(),
				pv.ParaverViewRuntimeTasks(),
				pv.ParaverViewTaskLabel(),
				pv.ParaverViewTaskSource(),
				pv.ParaverViewTaskId(),
				pv.ParaverViewHardwareCounters(),
				pv.ParaverViewThreadId(),
				pv.ParaverViewRuntimeSubsystems(),
				pv.ParaverViewCTFFlush(),
				#pv.ParaverViewNumberOfReadyTasks(),
				pv.ParaverViewNumberOfCreatedTasks(),
				pv.ParaverViewNumberOfBlockedTasks(),
				pv.ParaverViewNumberOfRunningTasks(),
				pv.ParaverViewNumberOfCreatedThreads(),
				pv.ParaverViewNumberOfRunningThreads(),
				pv.ParaverViewNumberOfBlockedThreads(),

				pv.ParaverViewKernelThreadID(),
				pv.ParaverViewKernelPreemptions(),
				pv.ParaverViewKernelSyscalls(),
			]

			# Install event processing hooks
			self.installHooks(KernelModel.preHooks())
			self.installHooks(RuntimeModel.preHooks())
			for view in self.__paraverViews:
				self.installHooks(view.hooks())
			self.installHooks(RuntimeModel.postHooks())
			self.installHooks(KernelModel.postHooks())

			# redirect message processing
			self.__process_message = self._process_other_message

			# compute set of starting events
			for view in self.__paraverViews:
				view.start(self.__payload)
			# We emit a set of initial events one nanosecond before the first
			# event is encountered to avoid overlapping with the extrae events
			# derived from the first ctf event.
			ParaverTrace.emitEvent(ts, 0, self.__payload)
			self.__payload.clear()

		self._consume_message(msg)

	def installHooks(self, hooks):
		for (name, func) in hooks:
			funcList = self.__hooks[name]
			funcList.append(func)

	def _process_other_message(self):
		msg = next(self._it)
		self._consume_message(msg)

	def _consume_message(self, msg):
		if self.__exitHandler.exit():
			# TODO exit using babeltrace2 API
			atexit.unregister(self._finalize)
			self._finalize()
			print("Conversion aborted successfully, trace might be incomplete but valid")
			sys.stdout.flush()
			os._exit(1)

		if type(msg) is bt2._EventMessageConst:
			ts = msg.default_clock_snapshot.value
			cpuId = msg.event["cpu_id"]

			if self.__verbose:
				name = msg.event.name
				print("event {}, cpu_id {}, timestamp {}".format(name, cpuId, ts))

			self._process_event(ts, cpuId, msg.event)
			self.__last = msg
		elif type(msg) is bt2._StreamBeginningMessageConst:
			# TODO use this to obtain env parameters
			#domain = msg.stream.trace.environment['domain']
			pass
		elif type(msg) is bt2._StreamEndMessageConst:
			pass
		elif type(msg) is bt2._PacketBeginningMessageConst:
			pass
		elif type(msg) is bt2._PacketEndMessageConst:
			pass
		else:
			raise RuntimeError("Unhandled message type", type(msg))
		#input("")

@bt2.plugin_component_class
class stats(bt2._UserSinkComponent):
	def __init__(self, config, params, obj):
		self.__port = self._add_input_port("in")
		self.__stats = defaultdict(int)
		self.__total = 0
		self.__exitHandler = ExitHandler()
		atexit.register(self._finalize)

	def _finalize(self):
		# sort dictionary
		sstats = sorted(self.__stats.items(), key=operator.itemgetter(1))
		# calc max padding for number of events
		padding = len(str(self.__total))
		# calc max padding for percentage
		ppmax = len(str(int((sstats[-1][1]/self.__total)*100)))

		# pretty print values
		for (key, value) in sstats:
			vpad = str(value).rjust(padding)
			per  = int((value/self.__total)*100)
			ppad = str(per).rjust(ppmax)
			print(vpad + " (" + ppad + "%): " + key)

		print("--------------------------------------------------")
		print(str(self.__total) + ": total number of events")

	def _user_graph_is_configured(self):
		self._it = self._create_message_iterator(self.__port)

	def _process_event(self, event):
		self.__stats[event.name] += 1
		self.__total += 1

	def _user_consume(self):
		msg = next(self._it)

		if self.__exitHandler().exit():
			# TODO exit using babeltrace2 API
			atexit.unregister(self._finalize)
			self._finalize()
			sys.stdout.flush()
			os._exit(1)

		if type(msg) is bt2._EventMessageConst:
			self._process_event(msg.event)
		elif type(msg) is bt2._StreamBeginningMessageConst:
			pass
		elif type(msg) is bt2._StreamEndMessageConst:
			pass
		elif type(msg) is bt2._PacketBeginningMessageConst:
			pass
		elif type(msg) is bt2._PacketEndMessageConst:
			pass
		else:
			raise RuntimeError("Unhandled message type", type(msg))

#### debug functions helper

def getMethods(obj):
	pprint([x for x in dir(obj) if callable(getattr(obj,x))])

def getProperties(obj):
	pprint([x for x in dir(obj) if not callable(getattr(obj,x))])

#variables = globals().copy()
#variables.update(locals())
#shell = code.InteractiveConsole(variables)
#shell.interact()
