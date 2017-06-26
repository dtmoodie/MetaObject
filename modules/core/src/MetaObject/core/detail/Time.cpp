#include "Time.hpp"
#include <chrono>

namespace mo{
    GetTime_f time_source = nullptr;
    MO_EXPORTS mo::Time_t getCurrentTime(){
        if (time_source) {
            return time_source();
        }
        auto ts = std::chrono::high_resolution_clock::now();
        return mo::Time_t(mo::ns * ts.time_since_epoch().count());
    }
    MO_EXPORTS void setTimeSource(GetTime_f timefunc){
        time_source = timefunc;
    }
}