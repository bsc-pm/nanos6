/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include "support/config/ConfigChecker.hpp"


std::string ConfigChecker::_initConditions;
bool ConfigChecker::_initChecked = false;
SpinLock ConfigChecker::_lock;
