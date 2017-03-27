#ifndef INSTRUMENT_HARDWARE_PLACE_ID_HPP
#define INSTRUMENT_HARDWARE_PLACE_ID_HPP


namespace Instrument {
	class hardware_place_id_t {
	public:
		typedef unsigned int inner_type_t;
		
	private:
		inner_type_t _id;
		
	public:
		hardware_place_id_t()
			: _id(~0)
		{
		}
		
		hardware_place_id_t(inner_type_t id)
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
		
		bool operator<(hardware_place_id_t other)
		{
			return (_id < other._id);
		}
		
	};
}


#endif // INSTRUMENT_HARDWARE_PLACE_ID_HPP
