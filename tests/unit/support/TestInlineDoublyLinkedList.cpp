/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include "TestAnyProtocolProducer.hpp"
#include "support/InlineDoublyLinkedList.hpp"

#include <algorithm>
#include <sstream>
#include <string>


struct MyListNode {
	int _value;
	InlineDoublyLinkedListLinks _links;
	
	MyListNode(int value)
		: _value(value), _links()
	{
	}
	
};


typedef InlineDoublyLinkedList<MyListNode, &MyListNode::_links> list_t;


template <typename T1, typename T2>
size_t getDistance(T1 first, T2 end) {
	size_t distance = 0;
	
	while (first != end) {
		first++;
		distance++;
	}
	
	return distance;
}


static std::string contentsToString(list_t const &myList) {
	std::ostringstream oss;
	
	oss << "Contents:";
	for (MyListNode const *node : myList) {
		oss << " " << node->_value;
	}
	
	return oss.str();
}


int main(__attribute__((unused)) int argc, __attribute__((unused)) char **argv) {
	TestAnyProtocolProducer tap;
	
	tap.registerNewTests(29);
	tap.begin();
	
	list_t myList;
	
	// 1
	tap.evaluate(myList.empty(), "the list is initially empty");
	tap.emitDiagnostic(contentsToString(myList));
	
	// 2
	MyListNode node1(1);
	myList.push_front(&node1);
	tap.evaluate(!myList.empty(), "after pushing an element to the front, the list becomes no longer empty");
	tap.emitDiagnostic(contentsToString(myList));
	
	// 3
	list_t::iterator it = myList.begin();
	tap.evaluate((*it)->_value == 1, "the beginning points to the correct element");
	tap.emitDiagnostic(contentsToString(myList));
	
	// 4
	list_t::reverse_iterator rit = myList.rbegin();
	tap.evaluate((*rit)->_value == 1, "the reverse beginning points to the correct element");
	tap.emitDiagnostic(contentsToString(myList));
	
	// 5 and 6
	tap.evaluate(getDistance(myList.begin(), myList.end()) == 1, "the size of the list is 1 using forward iterators");
	tap.emitDiagnostic(contentsToString(myList));
	tap.evaluate(getDistance(myList.rbegin(), myList.rend()) == 1, "the size of the list is 1 using backwards iterators");
	tap.emitDiagnostic(contentsToString(myList));
	
	// 7
	myList.erase(it);
	tap.evaluate(myList.empty(), "after removing the element through its iterator the list becomes empty");
	tap.emitDiagnostic(contentsToString(myList));
	
	// 8 and 9
	tap.evaluate(myList.begin() == myList.end(), "forward iterators indicate that the list is empty");
	tap.emitDiagnostic(contentsToString(myList));
	tap.evaluate(myList.rbegin() == myList.rend(), "backwards iterators indicate that the list is empty");
	tap.emitDiagnostic(contentsToString(myList));
	
	// 10
	MyListNode node2(2);
	myList.push_back(&node2);
	tap.evaluate(!myList.empty(), "after pushing an element to the back, the list becomes no longer empty");
	tap.emitDiagnostic(contentsToString(myList));
	
	// 11
	it = myList.begin();
	tap.evaluate((*it)->_value == 2, "the beginning points to the correct element");
	tap.emitDiagnostic(contentsToString(myList));
	
	// 12
	rit = myList.rbegin();
	tap.evaluate((*rit)->_value == 2, "the reverse beginning points to the correct element");
	tap.emitDiagnostic(contentsToString(myList));
	
	// 13 and 14
	tap.evaluate(myList.front()->_value == 2, "the front contains the correct element");
	tap.emitDiagnostic(contentsToString(myList));
	tap.evaluate(myList.back()->_value == 2, "the back contains the correct element");
	tap.emitDiagnostic(contentsToString(myList));
	
	// 15, 16 and 17
	it = myList.insert(myList.begin(), &node1);
	tap.evaluate((*it)->_value == 1, "inserting an element at the beginning returns an iterator that points to it");
	tap.emitDiagnostic(contentsToString(myList));
	it++;
	tap.evaluate((*it)->_value == 2, "the following element is the expected one");
	tap.emitDiagnostic(contentsToString(myList));
	it++;
	tap.evaluate(it == myList.end(), "the following position is the end of the list");
	tap.emitDiagnostic(contentsToString(myList));
	
	// 18, 19 and 20
	rit = myList.rbegin();
	tap.evaluate((*rit)->_value == 2, "the first element of a reverse iteration is the expected one");
	tap.emitDiagnostic(contentsToString(myList));
	rit++;
	tap.evaluate((*rit)->_value == 1, "the second element of a reverse iteration is the expected one");
	tap.emitDiagnostic(contentsToString(myList));
	rit++;
	tap.evaluate(rit == myList.rend(), "the next position is the reverse end");
	tap.emitDiagnostic(contentsToString(myList));
	
	it = myList.begin();
	it++;
	MyListNode node3(3);
	list_t::iterator it2 = myList.insert(it, &node3);
	
	// 21, 22 and 23
	tap.evaluate((*it2)->_value == 3, "inserting an element between two others adds returns an iterator that points to the new element");
	tap.emitDiagnostic(contentsToString(myList));
	it2--;
	tap.evaluate((*it2)->_value == 1, "the preceeding element is the expected one");
	tap.emitDiagnostic(contentsToString(myList));
	it2++; it2++;
	tap.evaluate((*it2)->_value == 2, "the following element is the expected one");
	tap.emitDiagnostic(contentsToString(myList));
	
	// 24
	it2++;
	tap.evaluate(it2 == myList.end(), "the following position is the end");
	tap.emitDiagnostic(contentsToString(myList));
	
	// 25
	MyListNode node4(4);
	myList.push_back(&node4);
	it = myList.end();
	it--;
	tap.evaluate((*it)->_value == 4, "adding an element to the end puts it in the correct position");
	tap.emitDiagnostic(contentsToString(myList));
	
#if 0
	// 26, 27, 28, 29 and 30
	std::reverse(myList.begin(), myList.end());
	it = myList.begin();
	tap.evaluate((*it)->_value == 4, "reversing the list puts the last element in the first position");
	tap.emitDiagnostic(contentsToString(myList));
	it++;
	tap.evaluate((*it)->_value == 3, "the following element is the expected one");
	tap.emitDiagnostic(contentsToString(myList));
	it++;
	tap.evaluate((*it)->_value == 1, "the following element is the expected one");
	tap.emitDiagnostic(contentsToString(myList));
	it++;
	tap.evaluate((*it)->_value == 2, "the following element is the expected one");
	tap.emitDiagnostic(contentsToString(myList));
	it++;
	tap.evaluate(it == myList.end(), "the following position is the end");
	tap.emitDiagnostic(contentsToString(myList));
	
	// 31, 32, 33 and 34
	it = myList.begin();
	it++;
	myList.erase(it);
	it = myList.begin();
	tap.evaluate((*it)->_value == 4, "removing the second element leaves the correct element in the first position");
	tap.emitDiagnostic(contentsToString(myList));
	it++;
	tap.evaluate((*it)->_value == 1, "the following element is the expected one");
	tap.emitDiagnostic(contentsToString(myList));
	it++;
	tap.evaluate((*it)->_value == 2, "the following element is the expected one");
	tap.emitDiagnostic(contentsToString(myList));
	it++;
	tap.evaluate(it == myList.end(), "the following position is the end");
	tap.emitDiagnostic(contentsToString(myList));
#else
	// 26, 27, 28 and 29
	it = myList.begin();
	it++;
	myList.erase(it);
	it = myList.begin();
	tap.evaluate((*it)->_value == 1, "removing the second element leaves the correct element in the first position");
	tap.emitDiagnostic(contentsToString(myList));
	it++;
	tap.evaluate((*it)->_value == 2, "the following element is the expected one");
	tap.emitDiagnostic(contentsToString(myList));
	it++;
	tap.evaluate((*it)->_value == 4, "the following element is the expected one");
	tap.emitDiagnostic(contentsToString(myList));
	it++;
	tap.evaluate(it == myList.end(), "the following position is the end");
	tap.emitDiagnostic(contentsToString(myList));
#endif
	
	tap.end();
	
	return tap.hasFailed();
}

