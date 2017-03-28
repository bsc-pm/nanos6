#ifndef INSTRUMENT_NULL_HARDWARE_PLACE_ID_HPP
#define INSTRUMENT_NULL_HARDWARE_PLACE_ID_HPP


namespace Instrument {
	class hardware_place_id_t {
	public:
		hardware_place_id_t()
		{
		}
		
		template<typename T>
		hardware_place_id_t(__attribute__((unused)) T id)
		{
		}
		
		bool operator==(__attribute__((unused)) hardware_place_id_t const &other) const
		{
			return true;
		}
		
	};
}


#endif // INSTRUMENT_NULL_HARDWARE_PLACE_ID_HPP

