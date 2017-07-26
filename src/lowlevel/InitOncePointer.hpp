/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INIT_ONCE_POINTER_HPP
#define INIT_ONCE_POINTER_HPP


#include <atomic>
#include <cassert>


template <typename BaseType>
class InitOncePointer {
public:
	//! \brief Initializes a pointer atomically with a new object of a given class
	//! 
	//! \param[inout] pointer a reference to the pointer
	//! \param constructorArgs the arguments that must be passed to the constructor
	//! 
	//! \returns true if the call performed the initialization, and false if it was already initialized
	//! 
	//! Example:
	//! 	static void *initOnFirstUse = nullptr;
	//! 	InitOncePointer<std::string>::init(initOnFirstUse, "This string will be initialized once");
	template <typename... ConstructorArgTypes>
	static inline bool init(void * &pointer, ConstructorArgTypes... constructorArgs)
	{
		// Constants that we use to identify the initialization stages
		void * const uninitializedValue = nullptr;
		void * const initializingValue = (void *) ~0UL;
		
		// Fast check
		if (__builtin_expect( (pointer != uninitializedValue) && (pointer != initializingValue), 1 )) {
			return false;
		}
		
		std::atomic<void *> &ref = (std::atomic<void *> &) (pointer);
		
		// Not initialized yet
		if (__builtin_expect(ref.load() == nullptr, 0)) {
			void *expected = nullptr;
			void *want = initializingValue;
			
			// Attempt to become the initializer
			if (ref.compare_exchange_strong(expected, want)) {
				// Construct the object
				void *newValue = new BaseType(constructorArgs...);
				assert(newValue != nullptr);
				
				// Assign it
				ref.store(newValue);
				
				return true;
			}
		}
		assert(ref.load() != uninitializedValue);
		
		// At this point it is being initialized (by another thread), or is already initialized
		while (ref.load() == initializingValue) {
			// Spin until initialized
		}
		
		// At this point the pointer must have been fully initialized
		assert((ref.load() != uninitializedValue) && (ref.load() != initializingValue));
		
		return false;
	}
	
	
	//! \brief Initializes a pointer atomically with a new object of a given class
	//! 
	//! \param[inout] pointer a reference to the pointer
	//! \param constructorArgs the arguments that must be passed to the constructor
	//! 
	//! \returns true if the call performed the initialization, and false if it was already initialized
	//! 
	//! Example:
	//! 	static std::string *initOnFirstUse = nullptr;
	//! 	InitOncePointer<std::string>::init(initOnFirstUse, "This string will be initialized once");
	template <typename... ConstructorArgTypes>
	static inline bool init(BaseType * &pointer, ConstructorArgTypes... constructorArgs)
	{
		return init((void * &) pointer, constructorArgs...);
	}
	
	
};



#endif // INIT_ONCE_POINTER_HPP
