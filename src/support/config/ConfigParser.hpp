/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef CONFIG_PARSER_HPP
#define CONFIG_PARSER_HPP

#include <cassert>
#include <dlfcn.h>
#include <stdexcept>
#include <string>
#include <vector>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#include "toml.hpp"
#pragma GCC diagnostic pop

#include "lowlevel/FatalErrorHandler.hpp"

class ConfigParser {
	toml::value data;
	toml::value empty = toml::value();

	toml::value findKey(std::string key)
	{
		std::string tmp;
		std::istringstream ss(key);
		toml::value *it = &data;

		while (std::getline(ss, tmp, '.')) {
			toml::value &current = *it;
			if (!current.is_table())
				return toml::value();

			if (!current.contains(tmp))
				return toml::value();

			toml::value &next = toml::find(current, tmp);
			it = &next;
		}

		return *it;
	}

public:
	ConfigParser()
	{
		const char *_nanos6_config_path = (const char *)dlsym(nullptr, "_nanos6_config_path");
		assert(_nanos6_config_path != nullptr);

		try {
			data = toml::parse(_nanos6_config_path);
		} catch (std::runtime_error &error) {
			FatalErrorHandler::fail("Error while opening the configuration file found in ",
				std::string(_nanos6_config_path),
				". Inner error: ",
				error.what());
		} catch (toml::syntax_error &error) {
			FatalErrorHandler::fail("Configuration syntax error: ", error.what());
		}
	}

	template <typename T>
	void get(std::string key, T &value, bool &found)
	{
		toml::value element = findKey(key);

		if (element.is_uninitialized()) {
			found = false;
			return;
		}

		try {
			value = toml::get<T>(element);
		} catch (toml::type_error &error) {
			found = false;

			FatalErrorHandler::warn(
				"Expecting type ", typeid(T).name(), " in configuration key ",
				key,
				", but found ",
				toml::stringize(element.type()),
				" instead.");
		}

		found = true;
	}

	template <typename T>
	void getList(std::string key, std::vector<T> &value, bool &found)
	{
		toml::value element = findKey(key);

		if (element.is_uninitialized()) {
			found = false;
			return;
		}

		try {
			value = toml::get<std::vector<T>>(element);
		} catch (toml::type_error &error) {
			found = false;

			FatalErrorHandler::warn(
				"Expecting type list(", typeid(T).name(), ") in configuration key ",
				key,
				", but found ",
				toml::stringize(element.type()),
				" instead.");
		}

		found = true;
	}
};

#endif // CONFIG_PARSER_HPP
