/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef CONTAINERS_HPP
#define CONTAINERS_HPP

#include <deque>
#include <map>
#include <queue>
#include <set>
#include <stack>
#include <unordered_map>
#include <vector>

#include <MemoryAllocator.hpp>


//! STL containers that leverage our memory allocators
namespace Container {

template <typename T>
using deque = std::deque<T, TemplateAllocator<T>>;

template <typename T>
using vector = std::vector<T, TemplateAllocator<T>>;

template <typename K, typename T, typename Compare = std::less<K>>
using map = std::map<K, T, Compare, TemplateAllocator<std::pair<const K, T>>>;

template <typename T, typename BackingContainer = vector<T>, typename Compare = std::less<T>>
using priority_queue = std::priority_queue<T, BackingContainer, Compare>;

template <typename T, typename BackingContainer = deque<T>>
using queue = std::queue<T, BackingContainer>;

template <typename T, typename Compare = std::less<T>>
using set = std::set<T, Compare, TemplateAllocator<T>>;

template <typename T, typename BackingContainer = deque<T>>
using stack = std::stack<T, BackingContainer>;

template <typename K, typename T, typename Hash = std::hash<K>, typename KeyEqual = std::equal_to<K>>
using unordered_map = std::unordered_map<K, T, Hash, KeyEqual, TemplateAllocator<std::pair<const K, T>>>;

} // namespace Container


#endif // CONTAINERS_HPP
