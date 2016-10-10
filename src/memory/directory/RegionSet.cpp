#include "RegionSet.hpp"

RegionSet::iterator RegionSet::begin(){
    return _set.begin();
}

RegionSet::iterator RegionSet::end(){
    return _set.end();
}

RegionSet::iterator RegionSet::find(void *address){
    return _set.find(address);
}   
