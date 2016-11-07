#ifndef CACHE_OBJECT_HPP
#define CACHE_OBJECT_HPP

#include <DataAccessRange.hpp>

#include "CopyObjectLinkingArtifacts.hpp"

#include "memory/Globals.hpp"

class CopyObject {
private: 
	DataAccessRange _range;
	unsigned int _version;
	cache_mask _caches;

public:
	
	CopyObjectLinkingArtifacts::hook_type _hook;

	CopyObject(void *startAddress, void *endAddress);
	CopyObject(void *startAddress, size_t size);

	DataAccessRange &getAccessRange();
	DataAccessRange const &getAccessRange() const;

	void *getStartAddress();
	void setStartAddress(void * startAddress);

	void *getEndAddress();
    void setEndAddress(void *endAddress);

	size_t getSize();

	int getVersion();
	void setVersion(int version);
	void incrementVersion();
	
	void addCache(int id);
	void removeCache(int id);
	bool testCache(int id);
	bool anyCache();
	int countCaches();
};

#endif //CACHE_OBJECT_HPP
