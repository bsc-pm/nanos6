#ifndef INSTRUMENT_VERBOSE_DATA_ACCESS_SEQUENCE_ID_HPP
#define INSTRUMENT_VERBOSE_DATA_ACCESS_SEQUENCE_ID_HPP


namespace Instrument {
	class data_access_sequence_id_t {
	public:
		typedef long int inner_type_t;
		
	private:
		inner_type_t _id;
		
	public:
		data_access_sequence_id_t(inner_type_t id)
		: _id(id)
		{
		}
		
		data_access_sequence_id_t()
		: _id(-1)
		{
		}
		
		operator inner_type_t() const
		{
			return _id;
		}
	};
}

#endif // INSTRUMENT_VERBOSE_DATA_ACCESS_SEQUENCE_ID_HPP
