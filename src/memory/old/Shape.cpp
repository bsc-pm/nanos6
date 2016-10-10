#include "Shape.hpp"

void Shape::reshape(int dimensions, unsigned int *shape){
	assert( dimensions > _dimensions ); // This should NEVER be called if there is no change in dimensions
	assert( shape[dimensions - 1 ] == _shape[_dimensions - 1 ] ); // Should this be moved to an if? TODO check	

	_dimensions = dimensions;
	_shape = shape;
	_bytes = Shape::getBytes(_dimensions, _shape, _itemSize);
	_bytesLastDim = _bytes / shape[0];

}

void Shape::reshape(std::size_t bytes){
	assert( bytes % _bytesLastDim == 0 ); // Cannot add a half dimension	

	_bytes += bytes;        
	_shape[0] += bytes / _bytesLastDim;

}
