#include <string>
#include <ostream>

#ifndef DATA_ACCESS_REGION_HPP
#define DATA_ACCESS_REGION_HPP

/*
 *  This is a placeholder file, needed for the nanos6 verbose instrumentation to compile
 */

class DataAccessRegion;
inline std::ostream & operator<<(std::ostream &o, DataAccessRegion const &region);

class DataAccessRegion {
    public:

    int getSize() const {
        return 0;
    }

    void * getStartAddress() const {
        return NULL;
    }
    friend std::ostream & ::operator<<(std::ostream &o, DataAccessRegion const &region);
};

inline std::ostream & operator<<(std::ostream &o, __attribute__((unused)) DataAccessRegion const &region) {
    return o << '.';
}

#endif // DATA_ACCESS_REGION_HPP
