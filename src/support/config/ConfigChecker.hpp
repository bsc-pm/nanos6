/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef CONFIG_CHECKER_HPP
#define CONFIG_CHECKER_HPP


#include <cassert>

#include "lowlevel/SpinLock.hpp"
#include "support/StringSupport.hpp"
#include "support/config/ConfigCentral.hpp"
#include "support/config/ConfigParser.hpp"


//! Static class used to check the values of options at run-time
//! in string format. It implements the OmpSs-2 assert feature
class ConfigChecker {
private:
	typedef ConfigCentral::OptionKind OptionKind;

	//! Conditions after initialization
	static std::string _initConditions;

	//! Whether conditions were checked after initialization
	static bool _initChecked;

	//! Lock to safely access the configuration conditions
	static SpinLock _lock;

private:
	//! \brief Check a given condition of a config option
	//!
	//! \param option The option name
	//! \param value The value used to compare
	//! \param op The comparison operator
	//!
	//! \returns Whether the condition is correct
	template <typename T>
	static inline bool checkCondition(const std::string &option, const std::string &value, const std::string &op)
	{
		T expected;
		bool success = StringSupport::parse(value, expected, std::boolalpha);
		if (!success) {
			FatalErrorHandler::fail("Could not convert option ", option, ", value ", value, ", op ", op);
		}

		T actual;
		success = ConfigCentral::getOptionValue<T>(option, actual);
		assert(success);

		if (op == "==")
			return (actual == expected);
		if (op == "!=")
			return (actual != expected);
		if (op == ">")
			return (actual > expected);
		if (op == ">=")
			return (actual >= expected);
		if (op == "<")
			return (actual < expected);
		if (op == "<=")
			return (actual <= expected);

		FatalErrorHandler::fail("Wrong operator for option ", option);
		return false;
	}

	//! \brief Assert multiple conditions of config options in string format
	//!
	//! This function asserts multiple conditions and aborts the program if
	//! any of the conditions are incorrect
	//!
	//! \param conditions A string of comma-separated conditions
	static inline void assertConditions(const std::string &conditions)
	{
		ConfigParser::processOptions(conditions, /* are conditions */ true,
			[&](const std::string &option, const std::string &value, const std::string &op) {
				if (!ConfigCentral::existsOption(option)) {
					FatalErrorHandler::fail("Option ", option, " does not exist");
				}

				if (ConfigCentral::isOptionList(option)) {
					FatalErrorHandler::fail("Option ", option, " is a list options and cannot be asserted");
				}

				OptionKind type = ConfigCentral::getOptionKind(option);

				bool success = false;
				switch (type) {
					case OptionKind::bool_option:
						success = checkCondition<ConfigCentral::bool_t>(option, value, op);
						break;
					case OptionKind::float_option:
						success = checkCondition<ConfigCentral::float_t>(option, value, op);
						break;
					case OptionKind::integer_option:
						success = checkCondition<ConfigCentral::integer_t>(option, value, op);
						break;
					case OptionKind::string_option:
						success = checkCondition<ConfigCentral::string_t>(option, value, op);
						break;
					case OptionKind::memory_option:
						success = checkCondition<ConfigCentral::memory_t>(option, value, op);
						break;
					default:
						assert(false);
						break;
				}

				if (!success) {
					FatalErrorHandler::fail("Assertion failed: ", option, op, value);
				}
			}
		);
	}

public:
	//! \brief Add conditions to assert
	//!
	//! This function adds conditions to assert. The conditions are
	//! asserted immediately after initializing the runtime, so it
	//! may return without having checked them yet. Conditions added
	//! after the initialization are asserted immediately
	static inline void addAssertConditions(const std::string &conditions)
	{
		assert(!conditions.empty());

		_lock.lock();
		if (!_initChecked) {
			_initConditions += "," + conditions;
			_lock.unlock();
		} else {
			_lock.unlock();
			assertConditions(conditions);
		}
	}

	//! \brief Assert the conditions registered before initialization
	//!
	//! This function should be called immediately after the runtime
	//! initialization and after having registered all config options
	//! in the config central
	static inline void assertConditions()
	{
		bool check = false;

		_lock.lock();
		if (!_initChecked) {
			check = true;
			_initChecked = true;
		}
		_lock.unlock();

		if (check)
			assertConditions(_initConditions);
	}
};


#endif // CONFIG_CHECKER_HPP
