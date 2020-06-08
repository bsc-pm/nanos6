from datetime import datetime

class ExtraeEventTypes():
	CTF_FLUSH                 = 9200009
	RUNTIME_CODE              = 9200010
	RUNTIME_BUSYWAITING       = 9200011
	RUNTIME_TASKS             = 9200012
	RUNNING_TASK_LABEL        = 9200013
	RUNNING_TASK_SOURCE       = 9200014
	RUNNING_THREAD_TID        = 9200015
	RUNNING_TASK_ID           = 9200016
	RUNTIME_SUBSYSTEMS        = 9200017
	NUMBER_OF_READY_TASKS     = 9200018
	NUMBER_OF_CREATED_TASKS   = 9200019
	NUMBER_OF_BLOCKED_TASKS   = 9200020
	NUMBER_OF_RUNNING_TASKS   = 9200021
	NUMBER_OF_CREATED_THREADS = 9200022
	NUMBER_OF_RUNNING_THREADS = 9200023
	NUMBER_OF_BLOCKED_THREADS = 9200024

class ExtraeEvent:
	def __init__(self, identifier, description, mid = 0, used = True):
		self.__id = identifier
		self.__description = description
		self.__values = {}
		self.__mid = mid
		self.__used = used

	def addValues(self, data):
		self.__values.update(data)

	def getExtraeId(self):
		return self.__id

	def isUsed(self):
		return self.__used

	def setUsed(self):
		self.__used = True

	def getStringHeader(self):
		return str(self.__mid) + "    " + str(self.__id) + "    " + self.__description + "\n"

	def __str__(self):
		entry  = "EVENT_TYPE\n"
		entry += self.getStringHeader()
		if self.__values:
			entry += "VALUES\n"
			for key in sorted(self.__values):
				entry += str(key) + "     " + str(self.__values[key]) + "\n"
		return entry

class ExtraeEventCollection:
	def __init__(self, startId, mid):
		self.__startId = startId
		self.__events = {}
		self.__mid = mid

	def addEvents(self, events):
		for (name, extraeId, desc) in events:
			self.__events.update({name : ExtraeEvent(extraeId, desc, self.__mid, used = False)})

	def addUnknownEvent(self, events):
		tmpId = self._getTemporalId()
		self.__events.update({name : ExtraeEvent(tmpId, name + " [Unknown]")})
		return tmpId

	def getExtraeId(self, name):
		event = self.__events[name]
		event.setUsed()
		return event.getExtraeId()

	def _getTemporalId(self):
		tmpId = self.__startId
		self.__startId += 1
		return tmpId

	def __str__(self):
		entry = ""
		for event in self.__events.values():
			if event.isUsed():
				entry += event.getStringHeader()
		if entry != "":
			entry = "EVENT_TYPE\n" + entry + "\n"
		return entry

class ParaverTrace:
	__name = "trace"
	__events = {}
	__ncpus = 0
	__nvcpus = 1 # leader thread virtual cpu
	__startTime = 0
	__endTime = 0
	__eventCollections = []
	__paraverHeaderDurationSlotSize = len(str(2**64))
	__paraverApplicationModeSlotSize = len(str(2**64))

	@classmethod
	def initalizeTraceFiles(cls):
		cls.__prvFile = open("./" + cls.__name + ".prv", "w")
		cls.__printParaverHeader(fake = True)

	@classmethod
	def addEventTypeAndValue(cls, extraeEventType, data, description = ""):
		if not extraeEventType in cls.__events:
			extraeEvent = ExtraeEvent(extraeEventType, description)
			cls.__events[extraeEventType] = extraeEvent
		else:
			extraeEvent = cls.__events[extraeEventType]

		extraeEvent.addValues(data)

	@classmethod
	def addEventType(cls, extraeEventType, description = ""):
		if not extraeEventType in cls.__events:
			extraeEvent = ExtraeEvent(extraeEventType, description)
			cls.__events[extraeEventType] = extraeEvent

	@classmethod
	def addEventCollection(cls, eventCollection):
		cls.__eventCollections.append(eventCollection)

	@classmethod
	def increaseVirtualCPUCount(cls):
		cls.__nvcpus += 1

	@classmethod
	def emitEvent(cls, ts, cpuID, data):
		cpuID += 1 # Paraver does not like CpuID 0
		entry = "2:0:1:1:{}:{}".format(cpuID, ts)
		for (eventType, eventValue) in data:
			entry += ":" + str(eventType) + ":" + str(eventValue)
		cls.__prvFile.write(entry + "\n")

	@classmethod
	def finalizeTraceFiles(cls):
		cls.__printParaverHeader(fake = False)
		cls.__prvFile.close()
		cls.__print_pcf()
		cls.__print_row()

	@classmethod
	def __print_pcf(cls):
		pcfStr =                             \
		"DEFAULT_OPTIONS\n"                  \
		"\n"                                 \
		"LEVEL               THREAD\n"       \
		"UNITS               NANOSEC\n"      \
		"LOOK_BACK           100\n"          \
		"SPEED               1\n"            \
		"FLAG_ICONS          ENABLED\n"      \
		"NUM_OF_STATE_COLORS 1000\n"         \
		"YMAX_SCALE          37\n"           \
		"\n"                                 \
		"\n"                                 \
		"DEFAULT_SEMANTIC\n"                 \
		"\n"                                 \
		"THREAD_FUNC          State As Is\n" \
		"\n"                                 \
		"\n"                                 \
		"STATES_COLOR\n"                     \
		"0    {117,195,255}\n"               \
		"1    {0,0,255}\n"                   \
		"\n"                                 \
		"\n"

		for event in cls.__events.values():
			pcfStr += str(event) + "\n\n"

		for eventCollection in cls.__eventCollections:
			pcfStr += str(eventCollection) + "\n\n"

		pcfFile = open("./" + cls.__name + ".pcf", "w")
		pcfFile.write(pcfStr)
		pcfFile.close()

	@classmethod
	def __print_row(cls):
		tcpus = cls.__ncpus + cls.__nvcpus
		rowStr = "LEVEL NODE SIZE 1\nhostname\n\n"
		rowStr += "LEVEL THREAD SIZE " + str(tcpus) + "\n"

		for cpu in range(cls.__ncpus):
			rowStr += "CPU " + str(cpu) + "\n"

		rowStr += "LT VCPU " + str(cls.__ltcpu) + "\n"

		for cpu in range(cls.__nvcpus - 1):
			rowStr += "ET VCPU " + str(cls.__ncpus + 1 + cpu) + "\n"

		rowFile = open("./" + cls.__name + ".row", "w")
		rowFile.write(rowStr)
		rowFile.close()

	@classmethod
	def __printParaverHeader(cls, fake):
		# Time header
		timeHeader = datetime.utcfromtimestamp(cls.__absoluteStartTime).strftime('#Paraver (%d/%m/%y at %H:%M):')

		# Trace Duration
		if fake:
			totalTime = 0
		else:
			if cls.__endTime == 0:
				raise Exception("Error: No trace endtime specified, cannot calculate total duration")
			totalTime = cls.__endTime - cls.__startTime
			if len(str(totalTime)) > cls.__paraverHeaderDurationSlotSize:
				raise Exception("Error: This trace duration is too big for the current __paraverHeaderDurationSlotSize. Please, manually incrase __paraverHeaderDurationSlotSize")
		paraverTotalTime = str(totalTime).zfill(cls.__paraverHeaderDurationSlotSize)
		ftime = "{}_ns:".format(paraverTotalTime)

		# Resource Model
		resourceModel = "0:" # disabled

		# Application Model
		nAppl = 1
		nTasks = 1
		nThreadsAppl1 = cls.__ncpus + cls.__nvcpus # this is a hack
		nodeIdAppl1 = 1
		nThreadsAppl1WithPadding = str(nThreadsAppl1).zfill(cls.__paraverApplicationModeSlotSize)
		applicationModel = "{}:{}({}:{})".format(nAppl, nTasks, nThreadsAppl1WithPadding, nodeIdAppl1)

		# Final Header
		paraverHeader = timeHeader + ftime + resourceModel + applicationModel

		cls.__prvFile.seek(0)
		cls.__prvFile.write(paraverHeader + "\n")

	@classmethod
	def addTraceName(cls, traceName):
		cls.__name = traceName

	@classmethod
	def addStartTime(cls, startTime):
		cls.__startTime = startTime

	@classmethod
	def addEndTime(cls, endTime):
		cls.__endTime = endTime

	@classmethod
	def addNumberOfCPUs(cls, ncpus):
		cls.__ncpus = ncpus
		cls.__ltcpu = ncpus # by convention

	@classmethod
	def addAbsoluteStartTime(cls, absoluteStartTime):
		cls.__absoluteStartTime = absoluteStartTime
