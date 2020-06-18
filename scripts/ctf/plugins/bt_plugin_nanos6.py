#
#	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
#
#	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
#

import bt2
import atexit

import readline
import code
from pprint import pprint

from collections import defaultdict
from runtime import RuntimeModel
from paravertrace import ParaverTrace, ExtraeEventTypes
import paraverviews as pv

import operator

bt2.register_plugin(
	module_name=__name__,
	name="nanos6",
	description="Nanos6 CTF instrumentation backend utilities",
	author="Aleix Roca Nonell",
	license="GPL",
	version=(1, 0, 0),
)

@bt2.plugin_component_class
class ctf2prv(bt2._UserSinkComponent):
	def __init__(self, config, params, obj):
		print("holi")
		self.__port = self._add_input_port("in")
		self.__process_message = self._process_first_message
		self.__last = None
		atexit.register(self._finalize)
		self.__payload = []

		self.__hooks = defaultdict(list)
		self.__paraverViews = [
			pv.ParaverViewRuntimeCode(),
			pv.ParaverViewRuntimeBusyWaiting(),
			pv.ParaverViewRuntimeTasks(),
			pv.ParaverViewTaskLabel(),
			pv.ParaverViewTaskSource(),
			pv.ParaverViewHardwareCounters(),
			pv.ParaverViewThreadId(),
			pv.ParaverViewTaskId(),
			pv.ParaverViewRuntimeSubsystems(),
			#pv.ParaverViewReadyTasks(),
			pv.ParaverViewCreatedTasks(),
			pv.ParaverViewBlockedTasks(),
			pv.ParaverViewRunningTasks(),
			pv.ParaverViewCreatedThreads(),
			pv.ParaverViewRunningThreads(),
			pv.ParaverViewBlockedThreads(),
			pv.ParaverViewCTFFlush(),
		]

	def _finalize(self):
		ts = self.__last.default_clock_snapshot.value
		ParaverTrace.addEndTime(ts)
		ParaverTrace.finalizeTraceFiles()
		print("adieu")

	def _user_graph_is_configured(self):
		self._it = self._create_message_iterator(self.__port)

	def _process_event(self, ts, event):
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
			ParaverTrace.initalizeTraceFiles()

			# install event processing hooks
			RuntimeModel.initialize(ncpus)
			self.installHooks(RuntimeModel.hooks())
			for view in self.__paraverViews:
				self.installHooks(view.hooks())

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
		if type(msg) is bt2._EventMessageConst:
			ts = msg.default_clock_snapshot.value
			name = msg.event.name
			cpu_id = msg.event["cpu_id"]
			print("event {}, cpu_id {}, timestamp {}".format(name, cpu_id, ts))
			self._process_event(ts, msg.event)
			self.__last = msg
		elif type(msg) is bt2._StreamBeginningMessageConst:
			print("Stream beginning")
		elif type(msg) is bt2._StreamEndMessageConst:
			print("Stream end")
		elif type(msg) is bt2._PacketBeginningMessageConst:
			print("Packet beginning")
		elif type(msg) is bt2._PacketEndMessageConst:
			print("Packet end")
		else:
			raise RuntimeError("Unhandled message type", type(msg))
		#input("")

@bt2.plugin_component_class
class stats(bt2._UserSinkComponent):
	def __init__(self, config, params, obj):
		self.__port = self._add_input_port("in")
		self.__stats = defaultdict(int)
		self.__total = 0
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
