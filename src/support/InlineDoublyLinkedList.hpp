/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INLINE_DOUBLY_LINKED_LIST
#define INLINE_DOUBLY_LINKED_LIST


#include "support/ConstPropagator.hpp"


#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>


struct InlineDoublyLinkedListLinks {
	InlineDoublyLinkedListLinks *_previous;
	InlineDoublyLinkedListLinks *_next;
	
	// Objects that participate in an InlineDoublyLinkedList cannot be simply copied as is.
	// 1. The destination previous links could end up being dangling
	// 2. The source links would still be pointed by other elements
	// Instead use the movement variants of the constructor and assignment operator
	InlineDoublyLinkedListLinks(InlineDoublyLinkedListLinks const &other) = delete;
	InlineDoublyLinkedListLinks &operator=(InlineDoublyLinkedListLinks const &other) = delete;
	
	InlineDoublyLinkedListLinks()
		: _previous(this), _next(this)
	{
	}
	
	InlineDoublyLinkedListLinks(InlineDoublyLinkedListLinks &&other)
		: _previous(std::move(other._previous)), _next(std::move(other._next)) 
	{
		// Relink
		_previous->_next = this;
		_next->_previous = this;
		
		other.clear();
	}
	
	InlineDoublyLinkedListLinks &operator=(InlineDoublyLinkedListLinks &&other)
	{
		// Move
		_previous = std::move(other._previous);
		_next = std::move(other._next);
		
		// Relink
		_previous->_next = this;
		_next->_previous = this;
		
		other.clear();
		
		return *this;
	}
	
	bool isInList() const
	{
		assert( ((_previous == this) && (_next == this)) || ((_previous != this) && (_next != this)) );
		
		return (_previous != this);
	}
	
	void clear()
	{
		_previous = this;
		_next = this;
	}
	
};





template <class T, typename ConstPropagator<T, InlineDoublyLinkedListLinks>::type T::*LINKS> class InlineDoublyLinkedList;
template <class T, typename ConstPropagator<T, InlineDoublyLinkedListLinks>::type T::*LINKS> class InlineDoublyLinkedListIterator;
template <class T, typename ConstPropagator<T, InlineDoublyLinkedListLinks>::type T::*LINKS> class InlineDoublyLinkedListReverseIterator;




template <class T, typename ConstPropagator<T, InlineDoublyLinkedListLinks>::type T::*LINKS>
class InlineDoublyLinkedListIteratorBase
	: public std::iterator<std::bidirectional_iterator_tag, std::size_t, T **, T *>
{
protected:
	typedef typename ConstPropagator<T, InlineDoublyLinkedListLinks>::type QualifyedLinksType;
	
	QualifyedLinksType *_links;
	
	template <class T2, typename ConstPropagator<T2, InlineDoublyLinkedListLinks>::type T2::*LINKS2> friend class InlineDoublyLinkedList;
	template <class T2, typename ConstPropagator<T2, InlineDoublyLinkedListLinks>::type T2::*LINKS2> friend class InlineDoublyLinkedListIterator;
	template <class T2, typename ConstPropagator<T2, InlineDoublyLinkedListLinks>::type T2::*LINKS2> friend class InlineDoublyLinkedListReverseIterator;
	
	
	static constexpr size_t getLinksOffset(T *node)
	{
		return (size_t) &(node->*LINKS) - (size_t) node;
	}
	
	// WARNING: This is not valid for the sentinel.
	static constexpr T *getNodeFromLinks(InlineDoublyLinkedListLinks const *links)
	{
		return (T *) ((size_t) links - getLinksOffset(0));
	}
	
	explicit InlineDoublyLinkedListIteratorBase(QualifyedLinksType *links)
		: _links(links)
	{
	}
	
public:
	InlineDoublyLinkedListIteratorBase()
		: _links(0)
	{
	}
	
	template <class OTHER_T, typename ConstPropagator<OTHER_T, InlineDoublyLinkedListLinks>::type OTHER_T::*OTHER_LINKS>
	bool operator==(InlineDoublyLinkedListIteratorBase<OTHER_T, OTHER_LINKS> const &other) const
	{
		return (_links == other._links);
	}
	
	template <class OTHER_T, typename ConstPropagator<OTHER_T, InlineDoublyLinkedListLinks>::type OTHER_T::*OTHER_LINKS>
	bool operator!=(InlineDoublyLinkedListIteratorBase<OTHER_T, OTHER_LINKS> const &other) const
	{
		return (_links != other._links);
	}
	
};




template <class T, typename ConstPropagator<T, InlineDoublyLinkedListLinks>::type T::*LINKS>
class InlineDoublyLinkedListIterator
	: public InlineDoublyLinkedListIteratorBase<T, LINKS>
{
	typedef InlineDoublyLinkedListIteratorBase<T, LINKS> BaseType;
	
protected:
	explicit InlineDoublyLinkedListIterator(typename BaseType::QualifyedLinksType *links)
		: BaseType(links)
	{
	}
	
	template <class T2, typename ConstPropagator<T2, InlineDoublyLinkedListLinks>::type T2::*LINKS2> friend class InlineDoublyLinkedList;
	template <class T2, typename ConstPropagator<T2, InlineDoublyLinkedListLinks>::type T2::*LINKS2> friend class InlineDoublyLinkedListIterator;
	template <class T2, typename ConstPropagator<T2, InlineDoublyLinkedListLinks>::type T2::*LINKS2> friend class InlineDoublyLinkedListReverseIterator;
	
	
public:
	InlineDoublyLinkedListIterator()
		: BaseType()
	{
	}
	
	operator InlineDoublyLinkedListIterator<const T, LINKS>() const
	{
		return InlineDoublyLinkedListIterator<const T, LINKS>(BaseType::_links);
	}
	
	void swap(InlineDoublyLinkedListIterator &other)
	{
		std::swap(BaseType::_links, other._links);
	}
	
	T *operator*()
	{
		T *node = BaseType::getNodeFromLinks(BaseType::_links);
		return node;
	}
	
#if 0
	T *operator->()
	{
		T *node = BaseType::getNodeFromLinks(BaseType::_links);
		return node;
	}
#endif
	
	InlineDoublyLinkedListIterator &operator++()
	{
		BaseType::_links = BaseType::_links->_next;
		return *this;
	}
	
	InlineDoublyLinkedListIterator operator++(int)
	{
		typename BaseType::QualifyedLinksType *oldLinks = BaseType::_links;
		BaseType::_links = BaseType::_links->_next;
		return InlineDoublyLinkedListIterator(oldLinks);
	}
	
	InlineDoublyLinkedListIterator &operator--()
	{
		BaseType::_links = BaseType::_links->_previous;
		return *this;
	}
	
	InlineDoublyLinkedListIterator operator--(int)
	{
		typename BaseType::QualifyedLinksType *oldLinks = BaseType::_links;
		BaseType::_links = BaseType::_links->_previous;
		return InlineDoublyLinkedListIterator(oldLinks);
	}
	
};


template <class T, typename ConstPropagator<T, InlineDoublyLinkedListLinks>::type T::*LINKS>
class InlineDoublyLinkedListReverseIterator
	: public InlineDoublyLinkedListIterator<T, LINKS>
{
	typedef InlineDoublyLinkedListIteratorBase<T, LINKS> BaseType;
	
protected:
	explicit InlineDoublyLinkedListReverseIterator(typename BaseType::QualifyedLinksType *links)
		: InlineDoublyLinkedListIterator<T, LINKS>(links)
	{
	}
	
	template <class T2, typename ConstPropagator<T2, InlineDoublyLinkedListLinks>::type T2::*LINKS2> friend class InlineDoublyLinkedList;
	template <class T2, typename ConstPropagator<T2, InlineDoublyLinkedListLinks>::type T2::*LINKS2> friend class InlineDoublyLinkedListIterator;
	template <class T2, typename ConstPropagator<T2, InlineDoublyLinkedListLinks>::type T2::*LINKS2> friend class InlineDoublyLinkedListReverseIterator;
	
public:
	InlineDoublyLinkedListReverseIterator()
		: InlineDoublyLinkedListIterator<T, LINKS>()
	{
	}
	
	InlineDoublyLinkedListReverseIterator(InlineDoublyLinkedListIterator<T, LINKS> other)
		: InlineDoublyLinkedListIterator<T, LINKS>(other)
	{
	}
	
	operator InlineDoublyLinkedListReverseIterator<const T, LINKS>() const
	{
		return InlineDoublyLinkedListReverseIterator<const T, LINKS>(BaseType::_links);
	}
	
	operator InlineDoublyLinkedListIterator<T, LINKS>() const
	{
		return InlineDoublyLinkedListIterator<T, LINKS>(BaseType::_links);
	}
	
	InlineDoublyLinkedListReverseIterator &operator++()
	{
		BaseType::_links = BaseType::_links->_previous;
		return *this;
	}
	
	InlineDoublyLinkedListReverseIterator operator++(int)
	{
		typename BaseType::QualifyedLinksType *oldLinks = BaseType::_links;
		BaseType::_links = BaseType::_links->_previous;
		return InlineDoublyLinkedListReverseIterator(oldLinks);
	}
	
	InlineDoublyLinkedListReverseIterator &operator--()
	{
		BaseType::_links = BaseType::_links->_next;
		return *this;
	}
	
	InlineDoublyLinkedListReverseIterator operator--(int)
	{
		typename BaseType::QualifyedLinksType *oldLinks = BaseType::_links;
		BaseType::_links = BaseType::_links->_next;
		return InlineDoublyLinkedListReverseIterator(oldLinks);
	}
	
};



namespace std {
	template <
		class T1, typename ConstPropagator<T1, InlineDoublyLinkedListLinks>::type T1::*LINKS1,
		class T2, typename ConstPropagator<T2, InlineDoublyLinkedListLinks>::type T2::*LINKS2
	>
	void iter_swap(InlineDoublyLinkedListIterator<T1, LINKS1> it1, InlineDoublyLinkedListIterator<T2, LINKS2> it2)
	{
		// Swap links
		InlineDoublyLinkedListLinks tmp = *it1._links;
		*it1._links = *it2._links;
		*it2._links = tmp;
		
		// Relink back previous and next
		InlineDoublyLinkedListLinks *links1 = it1._links;
		links1->_previous->_next = links1;
		links1->_next->_previous = links1;
		
		InlineDoublyLinkedListLinks *links2 = it2._links;
		links2->_previous->_next = links2;
		links2->_next->_previous = links2;
	}
}


template <class T, typename ConstPropagator<T, InlineDoublyLinkedListLinks>::type T::*LINKS>
class InlineDoublyLinkedList {
	typedef typename ConstPropagator<T, InlineDoublyLinkedListLinks>::type LinksType;
	
	struct HeadNode: public LinksType {
		HeadNode()
			: LinksType()
		{
		}
	};
	
	HeadNode _sentinel;
	
	
public:
	InlineDoublyLinkedList()
		: _sentinel()
	{
	}
	
	typedef T value_type;
	
	typedef T *pointer;
	typedef T const *const_pointer;
	typedef T *reference;
	typedef T const *const_reference;
	
	typedef InlineDoublyLinkedListIterator<T, LINKS> iterator;
	typedef InlineDoublyLinkedListIterator<const T, LINKS> const_iterator;
	typedef InlineDoublyLinkedListReverseIterator<T, LINKS> reverse_iterator;
	typedef InlineDoublyLinkedListReverseIterator<const T, LINKS> const_reverse_iterator;
	
	typedef size_t size_type;
	typedef ptrdiff_t difference_type;
	
	typedef void allocator_type; // This is an inline list, it does not allocate anything at all
	
	
	const_iterator begin() const
	{
		return const_iterator(_sentinel._next);
	}
	iterator begin()
	{
		return iterator(_sentinel._next);
	}
	
	const_iterator end() const
	{
		return const_iterator(&_sentinel);
	}
	iterator end()
	{
		return iterator(&_sentinel);
	}
	
	const_reverse_iterator rbegin() const
	{
		return const_reverse_iterator(--end());
	}
	reverse_iterator rbegin()
	{
		return reverse_iterator(--end());
	}
	
	const_reverse_iterator rend() const
	{
		return const_reverse_iterator(--begin());
	}
	reverse_iterator rend()
	{
		return reverse_iterator(--begin());
	}
	
	const_iterator cbegin() const
	{
		return const_iterator(_sentinel._next);
	}
	const_iterator cend() const
	{
		return const_iterator(&_sentinel);
	}
	const_reverse_iterator crbegin() const
	{
		return const_reverse_iterator(end());
	}
	const_reverse_iterator crend() const
	{
		return const_reverse_iterator(begin());
	}
	
	const_reference front() const
	{
		return *begin();
	}
	reference front()
	{
		return *begin();
	}
	
	const_reference back() const
	{
		const_iterator it = end();
		--it;
		return *it;
	}
	reference back()
	{
		iterator it = end();
		--it;
		return *it;
	}
	
	
	bool empty() const
	{
		return (_sentinel._next == &_sentinel);
	}
	
	iterator insert(iterator position, T *node)
	{
		LinksType *links = &(node->*LINKS);
		LinksType *previous = position._links->_previous;
		
		links->_previous = previous;
		links->_next = position._links;
		previous->_next = links;
		position._links->_previous = links;
		
		return iterator(links);
	}
	
	void push_back(T *node)
	{
		insert(end(), node);
	}
	
	void push_front(T *node)
	{
		insert(begin(), node);
	}
	
	
	iterator erase(iterator position)
	{
		LinksType *links = position._links;
		LinksType *previous = links->_previous;
		LinksType *next = links->_next;
		
		previous->_next = next;
		next->_previous = previous;
		
		#ifndef NDEBUG
			links->_next = 0;
			links->_previous = 0;
		#endif
		
		return iterator(next);
	}
	
	void remove(T *node)
	{
		LinksType *links = &(node->*LINKS);
		erase(iterator(links));
	}
	
	
};


#endif // INLINE_DOUBLY_LINKED_LIST
