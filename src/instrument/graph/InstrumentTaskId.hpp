#ifndef INSTRUMENT_TASK_ID_HPP
#define INSTRUMENT_TASK_ID_HPP


namespace Instrument {
	class task_id_t {
	public:
		typedef int inner_type_t;
		
	private:
		inner_type_t _id;
	
	public:
		task_id_t()
			: _id(-1)
		{
		}
		
		task_id_t(inner_type_t id)
			: _id(id)
		{
		}
		
		operator inner_type_t() const
		{
			return _id;
		}
		
		bool operator==(inner_type_t other) const
		{
			return (_id == other);
		}
		
		bool operator!=(inner_type_t other) const
		{
			return (_id != other);
		}
		
		bool operator<(task_id_t other)
		{
			return (_id < other._id);
		}
		
	};
	
}

#endif // INSTRUMENT_TASK_ID_HPP
