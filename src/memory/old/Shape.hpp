#ifndef SHAPE_HPP
#define SHAPE_HPP
	
#include <cstddef> //std::size_t
#include <climits> // CHAR_BIT
#include <assert.h> // assert
#include <iostream>

class Shape{
    
        friend class MemoryObject;

private:

        //int _bits; //< number of bits of the shape
        std::size_t _itemSize; //< size of the elements in the object (calculated with sizeof)
		unsigned int _dimensions; //< number of dimensions
		unsigned int *_shape; //< number of elements in each dimension, last dimension in bytes

        std::size_t _bytes; //< number of bytes of the shape
		std::size_t _bytesLastDim; //< size of bytes of the elements in the last dimension, precalculated for reshaping ops

public:

        Shape(unsigned int dimensions, unsigned int *shape, std::size_t itemSize)
        : _dimensions(dimensions),
        _shape(shape),
		_itemSize(itemSize){
            _bytes = Shape::getBytes(_dimensions, _shape, itemSize);
			_bytesLastDim = _bytes / shape[0];
            //_bits = n_bytes * CHAR_BIT; // There may be more than 8 bits in a byte depending on the implementation
        }

        ~Shape(){
            delete [] _shape;
        }
	
		/* \brief Adds new dimensions to the shape
		 *  
		 * Reshapes the Shape by adding new dimensions to it. 
		 * Only the last dimension may be incomplete in the chunks represented by shapes, since they have to represent contiguous memory.
		 * Due to this we can assume that the new shape contains all the elements in the other dimensions.
		 *
		 * \param[in] dimension number of dimension on the new shape
		 * \param[in] shape number of elements in each dimension of the new shape
		 */
		void reshape(int dimensions, unsigned int *shape);
	
		/* \brief Adds a number of bytes to the shape. 
		 *	
		 * Add a number of chunks of bytes to the shape. Only the number of bytes is specified (not right or left) since shape doesnt track the base address.
		 * The number of bytes has to be a multiple of the size of the last dimension
		 * 
		 * \param[in] bytes number of bytes to add 
		 */
		void reshape(std::size_t bytes);
	
		/* \brief Calculates the total number of bytes in a shape-like data (array of number of elements in each dimension)
		 *
		 * \param[in] dimensions number of dimension on the shape
		 * \param[in] shape number of elements in each dimension
		 */
		static size_t getBytes(unsigned int dimensions, unsigned int *shape, std::size_t itemSize){
			int bytes = shape[dimensions-1] * itemSize;
			for(int i = dimensions - 2; i >= 0; i--){
				bytes = shape[i] * bytes;
			}

			return bytes;
		}	

		/* Printing in Unit tests */
		friend std::ostream &operator<<(std::ostream &os, const Shape &obj){
			os << "{ Shape ";
			for(int i = 0; i < obj._dimensions-1; i++){
				os << "["<< obj._shape[i] <<"]";
			}
			os << "[" << obj._shape[obj._dimensions-1] << " * " << obj._itemSize << "]";
			os << " -> " << obj._bytes << " bytes ";
			os << "}";
			return os;		
		}
};

#endif //SHAPE_HPP
