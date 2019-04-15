/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#include <cassert>

#include <Chrono.hpp>


TickConversionUpdater *TickConversionUpdater::_tickUpdater;
double ChronoArchDependentData::_tickConversionFactor;

void TickConversionUpdater::beginUpdate()
{
	assert(_tickUpdater != nullptr);
	
	_tickUpdater->_updateStarted = true;
	_tickUpdater->_c1.restart();
	gettimeofday(&_tickUpdater->_t1, NULL);
	_tickUpdater->_c1.start();
}

void TickConversionUpdater::finishUpdate()
{
	assert(_tickUpdater != nullptr);
	
	_tickUpdater->_c1.stop();
	gettimeofday(&_tickUpdater->_t2, NULL);
	double start = (_tickUpdater->_t1.tv_sec * 1000000.0) + (_tickUpdater->_t1.tv_usec);
	double end   = (_tickUpdater->_t2.tv_sec * 1000000.0) + (_tickUpdater->_t2.tv_usec);
	double rate  = (end - start) / ((double) (_tickUpdater->_c1.getAccumulated()));
	ChronoArchDependentData::_tickConversionFactor = rate;
	_tickUpdater->_updateStarted = false;
}

void TickConversionUpdater::initialize()
{
	// Create the singleton
	if (_tickUpdater == nullptr) {
		_tickUpdater = new TickConversionUpdater();
	}
	
	// Begin the first conversion factor computation
	beginUpdate();
	
	// Register a service that automatically updates the tick conversion factor
	nanos6_register_polling_service(
		"UpdateTickConversionFactorService",
		updateTickConversionFactor,
		nullptr
	);
}

void TickConversionUpdater::shutdown()
{
	// Unregister the service that updates the tick conversion factor
	nanos6_unregister_polling_service(
		"UpdateTickConversionFactorService",
		updateTickConversionFactor,
		nullptr
	);
	
	if (_tickUpdater != nullptr) {
		delete _tickUpdater;
	}
}

int TickConversionUpdater::updateTickConversionFactor(void *)
{
	assert(_tickUpdater != nullptr);
	
	if (!_tickUpdater->_updateStarted) {
		_tickUpdater->_updateFrequencyChrono.start();
		_tickUpdater->beginUpdate();
	}
	else {
		_tickUpdater->_updateFrequencyChrono.stop();
		
		// Every 100 ms update the factor
		if (((double) (_tickUpdater->_updateFrequencyChrono)) > 100000.0) {
			_tickUpdater->finishUpdate();
			_tickUpdater->_updateFrequencyChrono.restart();
		}
		
		_tickUpdater->_updateFrequencyChrono.start();
	}
	
	return false;
}

