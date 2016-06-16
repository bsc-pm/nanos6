#ifndef INSTRUMENT_TASK_ID_HPP
#define INSTRUMENT_TASK_ID_HPP


namespace Instrument {
	namespace Extrae {
		struct TaskInfo {
			void *_userCode;
			
			TaskInfo(void *userCode)
				: _userCode(userCode)
			{
			}
		};
	}
	
	typedef Extrae::TaskInfo task_id_t;
}

#endif // INSTRUMENT_TASK_ID_HPP
