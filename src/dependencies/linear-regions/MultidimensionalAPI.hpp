#ifndef MULTIDIMENSIONAL_API_HPP
#define MULTIDIMENSIONAL_API_HPP


#include <nanos6/multidimensional-dependencies.h>
#include <nanos6/dependencies.h>

#include "../DataAccessType.hpp"
#include "../MultidimensionalAPITraversal.hpp"


#ifdef NDEBUG
#define _AI_ inline __attribute((always_inline))
#else
#define _AI_ inline
#endif

#define _UU_ __attribute__((unused))


template <DataAccessType ACCESS_TYPE, bool WEAK, typename... TS>
_AI_ void register_data_access_base(
	void *handler, int symbolIndex, char const *regionText, void *baseAddress,
	long currentDimSize, long currentDimStart, long currentDimEnd
);

template<>
_AI_ void register_data_access_base<READ_ACCESS_TYPE, true>(
	void *handler, _UU_ int symbolIndex, _UU_ char const *regionText, void *baseAddress,
	_UU_ long currentDimSize, _UU_ long currentDimStart, long currentDimEnd
) {
	size_t start = (size_t) baseAddress;
	start += currentDimStart;
	nanos_register_weak_read_depinfo(handler, (void *) start, currentDimEnd - currentDimStart);
}

template<>
_AI_ void register_data_access_base<READ_ACCESS_TYPE, false>(
	void *handler, _UU_ int symbolIndex, _UU_ char const *regionText, void *baseAddress,
	_UU_ long currentDimSize, long currentDimStart, long currentDimEnd
) {
	size_t start = (size_t) baseAddress;
	start += currentDimStart;
	nanos_register_read_depinfo(handler, (void *) start, currentDimEnd - currentDimStart);
}

template<>
_AI_ void register_data_access_base<WRITE_ACCESS_TYPE, true>(
	void *handler, _UU_ int symbolIndex, _UU_ char const *regionText, void *baseAddress,
	_UU_ long currentDimSize, long currentDimStart, long currentDimEnd
) {
	size_t start = (size_t) baseAddress;
	start += currentDimStart;
	nanos_register_weak_write_depinfo(handler, (void *) start, currentDimEnd - currentDimStart);
}

template<>
_AI_ void register_data_access_base<WRITE_ACCESS_TYPE, false>(
	void *handler, _UU_ int symbolIndex, _UU_ char const *regionText, void *baseAddress,
	_UU_ long currentDimSize, long currentDimStart, long currentDimEnd
) {
	size_t start = (size_t) baseAddress;
	start += currentDimStart;
	nanos_register_write_depinfo(handler, (void *) start, currentDimEnd - currentDimStart);
}

template<>
_AI_ void register_data_access_base<READWRITE_ACCESS_TYPE, true>(
	void *handler, _UU_ int symbolIndex, _UU_ char const *regionText, void *baseAddress,
	_UU_ long currentDimSize, long currentDimStart, long currentDimEnd
) {
	size_t start = (size_t) baseAddress;
	start += currentDimStart;
	nanos_register_weak_readwrite_depinfo(handler, (void *) start, currentDimEnd - currentDimStart);
}

template<>
_AI_ void register_data_access_base<READWRITE_ACCESS_TYPE, false>(
	void *handler, _UU_ int symbolIndex, _UU_ char const *regionText, void *baseAddress,
	_UU_ long currentDimSize, long currentDimStart, long currentDimEnd
) {
	size_t start = (size_t) baseAddress;
	start += currentDimStart;
	nanos_register_readwrite_depinfo(handler, (void *) start, currentDimEnd - currentDimStart);
}

template<>
_AI_ void register_data_access_base<CONCURRENT_ACCESS_TYPE, false>(
	void *handler, _UU_ int symbolIndex, _UU_ char const *regionText, void *baseAddress,
	_UU_ long currentDimSize, long currentDimStart, long currentDimEnd
) {
	size_t start = (size_t) baseAddress;
	start += currentDimStart;
	nanos_register_concurrent_depinfo(handler, (void *) start, currentDimEnd - currentDimStart);
}


template <DataAccessType ACCESS_TYPE, bool WEAK, typename... TS>
static _AI_ void register_data_access_skip_next(
	void *handler, int symbolIndex, char const *regionText, void *baseAddress,
	long currentDimSize, long currentDimStart, long currentDimEnd,
	long nextDimSize, long nextDimStart, long nextDimEnd,
	TS... otherDimensions
);


template <DataAccessType ACCESS_TYPE, bool WEAK>
static _AI_ void register_data_access(
	void *handler, int symbolIndex, char const *regionText, void *baseAddress,
	long currentDimSize, long currentDimStart, long currentDimEnd
) {
	register_data_access_base<ACCESS_TYPE, WEAK>(handler, symbolIndex, regionText, baseAddress, currentDimSize, currentDimStart, currentDimEnd);
}

template <DataAccessType ACCESS_TYPE, bool WEAK, typename... TS>
static _AI_ void register_data_access(
	void *handler, int symbolIndex, char const *regionText, void *baseAddress,
	long currentDimSize, long currentDimStart, long currentDimEnd,
	TS... otherDimensions
) {
	if (currentDimensionIsContinuous(otherDimensions...)) {
		register_data_access_skip_next<ACCESS_TYPE, WEAK>(handler, symbolIndex, regionText, baseAddress,
			currentDimSize * getCurrentDimensionSize(otherDimensions...),
			currentDimStart * getCurrentDimensionSize(otherDimensions...),
			currentDimEnd * getCurrentDimensionSize(otherDimensions...),
			otherDimensions...
		);
	} else {
		size_t stride = getStride<>(otherDimensions...);
		char *currentBaseAddress = (char *) baseAddress;
		currentBaseAddress += currentDimStart * stride;
		
		for (long index = currentDimStart; index < currentDimEnd; index++) {
			register_data_access<ACCESS_TYPE, WEAK>(handler, symbolIndex, regionText, currentBaseAddress, otherDimensions...);
			currentBaseAddress += stride;
		}
	}
}


template <DataAccessType ACCESS_TYPE, bool WEAK, typename... TS>
static _AI_ void register_data_access_skip_next(
	void *handler, int symbolIndex, char const *regionText, void *baseAddress,
	long currentDimSize, long currentDimStart, long currentDimEnd,
	_UU_ long nextDimSize, _UU_ long nextDimStart, _UU_ long nextDimEnd,
	TS... otherDimensions
) {
	register_data_access<ACCESS_TYPE, WEAK, TS...>(handler, symbolIndex, regionText, baseAddress, currentDimSize, currentDimStart, currentDimEnd, otherDimensions...);
}


// The following is only for reductions

template <typename... TS>
static _AI_ void register_reduction_access_skip_next(
	int reduction_operation, int reduction_index,
	void *handler, int symbolIndex, char const *regionText, void *baseAddress,
	long currentDimSize, long currentDimStart, long currentDimEnd,
	long nextDimSize, long nextDimStart, long nextDimEnd,
	TS... otherDimensions
);


template <typename... TS>
static _AI_ void register_reduction_access(
	int reduction_operation, int reduction_index,
	void *handler, int symbolIndex, char const *regionText, void *baseAddress,
	long currentDimSize, long currentDimStart, long currentDimEnd
) {
	nanos_register_region_reduction_depinfo1(
		reduction_operation, reduction_index,
		handler, symbolIndex, regionText, baseAddress,
		currentDimSize, currentDimStart, currentDimEnd
	);
}

template <typename... TS>
static _AI_ void register_reduction_access(
	int reduction_operation, int reduction_index,
	void *handler, int symbolIndex, char const *regionText, void *baseAddress,
	long currentDimSize, long currentDimStart, long currentDimEnd,
	TS... otherDimensions
) {
	if (currentDimensionIsContinuous(otherDimensions...)) {
		register_reduction_access_skip_next(
			reduction_operation, reduction_index,
			handler, symbolIndex, regionText, baseAddress,
			currentDimSize * getCurrentDimensionSize(otherDimensions...),
			currentDimStart * getCurrentDimensionSize(otherDimensions...),
			currentDimEnd * getCurrentDimensionSize(otherDimensions...),
			otherDimensions...
		);
	} else {
		size_t stride = getStride<>(otherDimensions...);
		char *currentBaseAddress = (char *) baseAddress;
		currentBaseAddress += currentDimStart * stride;
		
		for (long index = currentDimStart; index < currentDimEnd; index++) {
			register_reduction_access<>(reduction_operation, reduction_index, handler, symbolIndex, regionText, baseAddress, otherDimensions...);
			currentBaseAddress += stride;
		}
	}
}


template <typename... TS>
static _AI_ void register_reduction_access_skip_next(
	int reduction_operation, int reduction_index,
	void *handler, int symbolIndex, char const *regionText, void *baseAddress,
	long currentDimSize, long currentDimStart, long currentDimEnd,
	_UU_ long nextDimSize, _UU_ long nextDimStart, _UU_ long nextDimEnd,
	TS... otherDimensions
) {
	register_reduction_access<TS...>(reduction_operation, reduction_index, handler, symbolIndex, regionText, baseAddress, currentDimSize, currentDimStart, currentDimEnd, otherDimensions...);
}

#undef _AI_
#undef _UU_


#endif // MULTIDIMENSIONAL_API_HPP
