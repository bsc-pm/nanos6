#ifndef INSTRUMENT_CPU_ID_HPP
#define INSTRUMENT_CPU_ID_HPP


namespace Instrument {
	class cpu_id_t {
	public:
		typedef unsigned int inner_type_t;
		
	private:
		inner_type_t _id;
		
	public:
		cpu_id_t()
			: _id(~0)
		{
		}
		
		cpu_id_t(inner_type_t id)
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
		
		bool operator<(cpu_id_t other)
		{
			return (_id < other._id);
		}
		
	};
}


#endif // INSTRUMENT_CPU_ID_HPP
