class ExecutionModel():
    """ This class caches common data to avoid repetitive event data fields
    lockups """

    _ts    = None
    _cpuid = None

    @classmethod
    def setCurrentEventData(cls, ts, cpuId):
        cls._ts = ts
        cls._cpuId = cpuId

    @classmethod
    def getCurrentCPUId(cls):
        return cls._cpuId

    @classmethod
    def getCurrentTimestamp(cls):
        return cls._ts
