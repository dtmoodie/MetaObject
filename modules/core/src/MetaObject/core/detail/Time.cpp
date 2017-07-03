#include "Time.hpp"
#include <chrono>
#include <ostream>
namespace mo{
    GetTime_f time_source = nullptr;
    MO_EXPORTS mo::Time_t getCurrentTime(){
        if (time_source) {
            return time_source();
        }
        return std::chrono::high_resolution_clock::now();
    }
    MO_EXPORTS void setTimeSource(GetTime_f timefunc){
        time_source = timefunc;
    }
    std::ostream& operator <<(std::ostream& lhs, const mo::Time_t& rhs){
        lhs << rhs.time_since_epoch().count() << " ns";
        return lhs;
    }
}

namespace std{
std::ostream& operator <<(std::ostream& lhs, const std::chrono::system_clock::time_point& rhs){
    lhs << rhs.time_since_epoch().count() << " ns";
    return lhs;
}

std::ostream& operator <<(std::ostream& lhs, const std::chrono::milliseconds& rhs){
    lhs << rhs.count() << " ms";
    return lhs;
}

std::ostream& operator <<(std::ostream& lhs, const std::chrono::microseconds& rhs){
    lhs << rhs.count() << " us";
    return lhs;
}

std::ostream& operator <<(std::ostream& lhs, const std::chrono::nanoseconds& rhs){
    lhs << rhs.count() << " ns";
    return lhs;
}

std::ostream& operator <<(std::ostream& lhs, const std::chrono::seconds& rhs){
    lhs << rhs.count() << " s";
    return lhs;
}
}
