#include "tasks/Task.hpp"

#include "DataAccess.hpp"

bool DataAccess::isSymbol(int index) const {
	assert(index < _originator->getSymbolNum());
	
	return _symbols[index];
}

void DataAccess::setSymbol(int index){
	assert(index < _originator->getSymbolNum());
	
	_symbols.set(index);
}

void DataAccess::unsetSymbol(int index){
	assert(index < _originator->getSymbolNum());
	
	_symbols.reset(index);
}	
	
