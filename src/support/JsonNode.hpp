/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef JSON_NODE_HPP
#define JSON_NODE_HPP

#include <string>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include "lowlevel/FatalErrorHandler.hpp"

namespace Json = boost::property_tree;


template <class T = double>
class JsonNode {
private:

	//! The inner ptree node
	Json::ptree _node;

public:

	inline JsonNode() :
		_node()
	{
	}

	inline JsonNode(const Json::ptree &node) :
		_node(node)
	{
	}

	inline JsonNode(Json::ptree &node) :
		_node(node)
	{
	}

	inline JsonNode(const JsonNode &node) :
		_node(node.getInnerNode())
	{
	}

	//! \brief Explicitely declare a default copy assignment to avoid
	//! warnings (gcc9+)
	JsonNode &operator=(const JsonNode &node) = default;

	inline const Json::ptree &getInnerNode() const
	{
		return _node;
	}


	//    NODE LEAF INTERFACE    //

	//! \brief Check if a certain data exists in this node
	//!
	//! \param[in] label The label of the data
	inline bool dataExists(const std::string &label) const
	{
		bool exists = false;

		// First check if the node exists
		if (childNodeExists(label)) {
			// Retreive the node and check if it is a terminal value
			Json::ptree childNode = getChildNode(label).getInnerNode();
			exists = (childNode.empty() && !childNode.data().empty());
		}

		return exists;
	}

	//! \brief Add data to this node
	//!
	//! \param[in] label The label for the data
	//! \param[in] data The data itself
	inline void addData(const std::string &label, const T &data)
	{
		_node.put(label, data);
	}

	//! \brief Get data from this node
	//!
	//! \param[in] label The label of the data
	//! \param[out] converted Whether the data extraction was successful
	template <typename X = T>
	inline X getData(const std::string &label, bool &converted) const
	{
		X data;
		converted = true;

		try {
			data = _node.get<X>(label);
		} catch (const Json::ptree_error &conversionError) {
			FatalErrorHandler::warn("Could not convert JSON data with label: ", label);
			converted = false;
		}

		return data;
	}


	//    NODE TREE INTERFACE    //

	//! \brief Delete everything in the tree
	inline void clear()
	{
		_node.clear();
	}

	//! \brief Check if a certain node exists
	//!
	//! \param[in] label The label of the node
	inline bool childNodeExists(const std::string &label) const
	{
		return (_node.find(label) != _node.not_found());
	}

	//! \brief Add a child node to this node
	//!
	//! \param[in] label The label of the node
	//! \param[in] node A JsonNode representing the node to add
	inline void addChildNode(const std::string &label, const JsonNode &node)
	{
		_node.push_back(Json::ptree::value_type(label, node.getInnerNode()));
	}

	//! \brief Get a child node of this node
	//!
	//! \param[in] label The label of the child node
	inline JsonNode getChildNode(const std::string &label) const
	{
		return JsonNode(_node.get_child(label));
	}

	//! \brief Traverse all direct child nodes of the inner node and
	//! apply a certain function in each of them
	//!
	//! \param[in] functionToApply The function to apply to each child node
	template <typename F>
	inline void traverseChildrenNodes(F functionToApply)
	{
		// Iterate all children nodes
		for (auto const &childNode : _node) {
			// Retreive the child node and its label
			const std::string &label = childNode.first;
			JsonNode<> node(childNode.second);

			functionToApply(label, node);
		}
	}

};

#endif // JSON_NODE_HPP
