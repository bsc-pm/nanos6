#include <cassert>
#include <map>
#include <mutex>
#include <sstream>
#include <string>

#include <nanos6/polling.h>

#include "PollingAPI.hpp"
#include "lowlevel/PaddedSpinLock.hpp"
#include "system/RuntimeInfo.hpp"


namespace PollingAPI {
	typedef PaddedSpinLock<128> lock_t;
	
	
	//! \brief the parameters of the nanos_register_polling_service function
	struct ServiceKey {
		std::string _name;
		nanos_polling_service_t _function;
		void *_functionData;
		
		ServiceKey(char const *name, nanos_polling_service_t function, void *functionData)
			: _name(name), _function(function), _functionData(functionData)
		{
		}
		
		ServiceKey(ServiceKey &&other)
			: _name(std::move(other._name)), _function(other._function), _functionData(other._functionData)
		{
		}
	};
	
	
	//! \brief the status of a service
	struct ServiceData {
		//! \brief Indicates whether the service is being processed at that moment
		bool _processing;
		
		//! \brief Indicates whether the service has been marked for removal
		bool _discard;
		
		ServiceData()
			: _processing(false), _discard(false)
		{
		}
	};
	
	
	// Container type for the services
	typedef std::map<ServiceKey, ServiceData> services_t;
	
	
	//! \brief This is held during traversal and modification operations but not while processing a service
	lock_t _lock;
	
	//! \brief Services in the system
	services_t _services;
	
	
	inline bool operator<(ServiceKey const &a, ServiceKey const &b)
	{
		if (a._function < b._function) {
			return true;
		} else if (a._function > b._function) {
			return false;
		}
		
		if (a._functionData < b._functionData) {
			return true;
		} else if (a._functionData > b._functionData) {
			return false;
		}
		
#if 0
		if (a._name < b._name) {
			return true;
		} else if (a._name > b._name) {
			return false;
		}
#endif
		
		return false; // Equal
	}
}


using namespace PollingAPI;


extern "C" void nanos_register_polling_service(char const *service_name, nanos_polling_service_t service_function, void *service_data)
{
	std::lock_guard<PollingAPI::lock_t> guard(PollingAPI::_lock);
	
	static std::map<nanos_polling_service_t, std::string> uniqueRegisteredServices;
	
	auto result = PollingAPI::_services.emplace(
		ServiceKey(service_name, service_function, service_data),
		ServiceData()
	);
	
	if (!result.second) {
		// The node was already added
		ServiceData &serviceData = result.first->second;
		
		// So it must have been marked as discarded
		assert(serviceData._discard);
		
		// Remove the mark
		serviceData._discard = false;
	} else {
		auto it = uniqueRegisteredServices.find(service_function);
		if (it == uniqueRegisteredServices.end()) {
			uniqueRegisteredServices[service_function] = service_name;
			std::ostringstream oss, oss2;
			oss << "registered_service_" << uniqueRegisteredServices.size();
			oss2 << "Registered Service " << uniqueRegisteredServices.size();
			
			RuntimeInfo::addEntry(oss.str(), oss2.str(), service_name);
		}
	}
}


extern "C" void nanos_unregister_polling_service(char const *service_name, nanos_polling_service_t service_function, void *service_data)
{
	ServiceKey key(service_name, service_function, service_data);
	
	std::lock_guard<PollingAPI::lock_t> guard(PollingAPI::_lock);
	auto it = PollingAPI::_services.find(key);
	
	assert((it != PollingAPI::_services.end()) && "Attempt to unregister a non-existing polling service");
	ServiceData &serviceData = it->second;
	
	assert(!serviceData._discard && "Attempt to unregister an already unregistered polling service");
	serviceData._discard = true;
}


void PollingAPI::handleServices()
{
	PollingAPI::_lock.lock();
	
	auto it = _services.begin();
	while (it != _services.end()) {
		ServiceKey const &serviceKey = it->first;
		ServiceData &serviceData = it->second;
		
		if (serviceData._processing) {
			// Somebody else processing it?
			it++;
			continue;
		}
		
		if (serviceData._discard) {
			it = _services.erase(it);
			continue;
		}
		
		// Set the processing flag
		assert(!serviceData._processing);
		serviceData._processing = true;
		
		// Execute the callback without locking
		PollingAPI::_lock.unlock();
		bool unregister = serviceKey._function(serviceKey._functionData);
		PollingAPI::_lock.lock();
		
		// Unset the processing flag
		assert(serviceData._processing);
		serviceData._processing = false;
		
		// By construction, even in the presence of concurrent calls to this method, the iterator remains valid
		
		// If the function returns true or the service had been marked for unregistration, remove the service
		if (unregister || serviceData._discard) {
			it = _services.erase(it);
		} else {
			it++;
		}
	}
	
	PollingAPI::_lock.unlock();
}

