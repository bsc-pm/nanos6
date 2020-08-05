#
#	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
#
#	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
#

from datetime import datetime

class ExtraeEventTypes():
	CTF_FLUSH                 = 6400009
	RUNTIME_CODE              = 6400010
	RUNTIME_BUSYWAITING       = 6400011
	RUNTIME_TASKS             = 6400012
	RUNNING_TASK_LABEL        = 6400013
	RUNNING_TASK_SOURCE       = 6400014
	RUNNING_THREAD_TID        = 6400015
	RUNNING_TASK_ID           = 6400016
	RUNTIME_SUBSYSTEMS        = 6400017
	NUMBER_OF_READY_TASKS     = 6400018
	NUMBER_OF_CREATED_TASKS   = 6400019
	NUMBER_OF_BLOCKED_TASKS   = 6400020
	NUMBER_OF_RUNNING_TASKS   = 6400021
	NUMBER_OF_CREATED_THREADS = 6400022
	NUMBER_OF_RUNNING_THREADS = 6400023
	NUMBER_OF_BLOCKED_THREADS = 6400024

	KERNEL_THREAD_ID          = 6400100
	KERNEL_PREEMPTIONS        = 6400101
	KERNEL_SYSCALLS           = 6400102

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

	def addUnknownEvent(self, name):
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
	__maxRealCPUId = 0
	__cpuIdIndex = 0
	__realCPUList = None

	__binaryName = "undefined"
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
		cls.__cpuMap.append(cls.__cpuIdIndex)
		cls.__cpuIdIndex += 1

	@classmethod
	def emitEvent(cls, ts, cpuId, data):
		cpuId = cls.__cpuMap[cpuId]
		entry = "2:0:1:1:{}:{}".format(cpuId, ts)
		for (eventType, eventValue) in data:
			entry += ":" + str(eventType) + ":" + str(eventValue)
		cls.__prvFile.write(entry + "\n")

	@classmethod
	def emitCommunicationEvent(cls, cpuSendId, timeSend, cpuRecvId, timeRecv):
		cpuSendId  = cls.__cpuMap[cpuSendId]
		cpuRecvId  = cls.__cpuMap[cpuRecvId]
		size       = 1
		tag        = 1

		objectSend = "0:1:1:{}".format(cpuSendId)
		objectRecv = "0:1:1:{}".format(cpuRecvId)
		entry = "3:{}:{}:{}:{}:{}:{}:{}:{}\n".format(
		        objectSend, timeSend, timeSend,
		        objectRecv, timeRecv, timeRecv,
		        size, tag)

		cls.__prvFile.write(entry)

	@classmethod
	def finalizeTraceFiles(cls):
		cls.__printParaverHeader(fake = False)
		cls.__prvFile.close()
		cls.__print_pcf()
		cls.__print_row()

	@classmethod
	def __buildStateColors(cls):

		# color definitions
		deepblue  = (  0,   0, 255)
		lightgrey = (217, 217, 217)
		red       = (230,  25,  75)
		green     = (60,  180,  75)
		yellow    = (255, 225,  25)
		orange    = (245, 130,  48)
		purple    = (145,  30, 180)
		cyan      = ( 70, 240, 240)
		magenta   = (240, 50,  230)
		lime      = (210, 245,  60)
		pink      = (250, 190, 212)
		teal      = (  0, 128, 128)
		lavender  = (220, 190, 255)
		brown     = (170, 110,  40)
		beige     = (255, 250, 200)
		maroon    = (128,   0,   0)
		mint      = (170, 255, 195)
		olive     = (128, 128,   0)
		apricot   = (255, 215, 180)
		navy      = (  0,   0, 128)
		blue      = (0,   130, 200)
		grey      = (128, 128, 128)
		black     = (  0,   0,   0)
		#white     = (255, 255, 255)
		#ompss2_1  = (107, 165, 217)
		#ompss2_2  = ( 43,  83, 160)

		# used colors
		colors = [
			black     , # (never shown anyways)
			deepblue  , # runtime
			lightgrey , # busy wait
			red       , # task
			green     ,
			yellow    ,
			orange    ,
			purple    ,
			cyan      ,
			magenta   ,
			lime      ,
			pink      ,
			teal      ,
			grey      ,
			lavender  ,
			brown     ,
			beige     ,
			maroon    ,
			mint      ,
			olive     ,
			apricot   ,
			navy      ,
			blue
		]

		cnt = 0
		strColors = "STATES_COLOR\n"
		for (r,g,b) in colors:
			strColors += "{:<3} {{{:>3}, {:>3}, {:>3}}}\n".format(cnt, r, g, b)
			cnt += 1

		strColors += "\n\n"

		return strColors


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
		"\n"

		pcfStr += cls.__buildStateColors()

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
			rowStr += "CPU " + str(cls.__realCPUList[cpu]) + "\n"

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
	def addCPUList(cls, cpuListStr):
		cpuList = [int(cpu) for cpu in cpuListStr.split(",")]
		cpuList.sort()

		# Detect max cpu Id
		maxCPUId = 0
		for cpu in cpuList:
			if cpu > maxCPUId:
				maxCPUId = cpu

		# Create cpu map from real system Id to Paraver Id
		cls.__cpuIdIndex = 1 # Paraver does not like cpu id 0
		cls.__cpuMap = [-1] * (maxCPUId + 1 + 1)
		for cpu in cpuList:
			cls.__cpuMap[cpu] = cls.__cpuIdIndex
			cls.__cpuIdIndex += 1
		# Add Leader Thread entry too
		cls.__cpuMap[maxCPUId + 1] = cls.__cpuIdIndex
		cls.__cpuIdIndex += 1

		cls.__ncpus = len(cpuList)
		cls.__maxRealCPUId = maxCPUId
		cls.__ltcpu = maxCPUId + 1 # by convention
		cls.__realCPUList = cpuList

	@classmethod
	def addBinaryName(cls, binaryName):
		cls.__binaryName = binaryName

	@classmethod
	def addAbsoluteStartTime(cls, absoluteStartTime):
		cls.__absoluteStartTime = absoluteStartTime

	@classmethod
	def getBinaryName(cls):
		return cls.__binaryName

	@classmethod
	def getNumberOfCPUs(cls):
		return cls.__ncpus

	@classmethod
	def getMaxRealCPUId(cls):
		return cls.__maxRealCPUId

	@classmethod
	def getLeaderThreadCPUId(cls):
		return cls.__ltcpu
