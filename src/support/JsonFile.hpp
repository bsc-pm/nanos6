/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef JSON_FILE_HPP
#define JSON_FILE_HPP

#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include "JsonNode.hpp"
#include "lowlevel/FatalErrorHandler.hpp"

namespace Json = boost::property_tree;


//! NOTE: Thread safety must be guaranteed as this class is not thread-safe
class JsonFile {

private:

	//! The path (name) of the system file this JsonFile mirrors
	std::string _path;

	//! Wether the mirrored system file exists
	bool _fileExists;

	//! The root node of the Json data
	JsonNode<> *_rootNode;

public:

	inline JsonFile(const std::string &path) :
		_path(path)
	{
		struct stat pathStatus;
		_fileExists = (stat(path.c_str(), &pathStatus) == 0);

		_rootNode = new JsonNode<>();
	}

	inline ~JsonFile()
	{
		if (_rootNode != nullptr) {
			delete _rootNode;
		}
	}

	inline const std::string &getPath() const
	{
		return _path;
	}

	inline JsonNode<> *getRootNode() const
	{
		return _rootNode;
	}

	inline bool fileExists() const
	{
		return _fileExists;
	}

	//! \brief Clear whatever data might be residing in the file
	inline void clearFile()
	{
		if (_rootNode != nullptr) {
			_rootNode->clear();
		}
	}

	//! \brief Populate the JsonFile with data from the real file it mirrors
	inline void loadData()
	{
		if (_fileExists) {
			// Load data from the file into the root node
			Json::ptree rootNode;
			try {
				Json::read_json(_path, rootNode);
			} catch (const Json::json_parser::json_parser_error &readError) {
				FatalErrorHandler::warn("JSON error when trying to load data from '", _path, "'");
				return;
			}

			assert(_rootNode != nullptr);

			// Initialize the root node with the identifier node
			*_rootNode = JsonNode<>(rootNode);
		}
	}

	//! \brief Store the JsonFile's data in the system file it mirrors
	inline void storeData() const
	{
		// Create/open a file
		std::ofstream file(_path.c_str());
		FatalErrorHandler::failIf(!file.is_open(), "Unable to create a file in '", _path, "'");

		// Write the data into the file
		assert(_rootNode != nullptr);
		Json::write_json(_path, _rootNode->getInnerNode());
		file.flush();
		file.close();
	}

};

#endif // JSON_FILE_HPP
