#include "Time.hpp"
#include <chrono>
#include <ostream>
#include <sstream>
#include <iomanip>

namespace mo
{
    GetTime_f time_source = nullptr;
    MO_EXPORTS mo::Time_t getCurrentTime()
    {
        if (time_source) 
        {
            return time_source();
        }
        return std::chrono::high_resolution_clock::now();
    }

    MO_EXPORTS void setTimeSource(GetTime_f timefunc)
    {
        time_source = timefunc;
    }

    MO_EXPORTS std::string printTime(mo::Time_t ns) 
    {
        typedef std::chrono::duration<int, std::ratio<86400>> days;
        std::stringstream ss;
        ss.fill('0');
        auto d = std::chrono::duration_cast<days>(ns.time_since_epoch());
        ns -= d;
        auto h = std::chrono::duration_cast<std::chrono::hours>(ns.time_since_epoch());
        ns -= h;
        auto m = std::chrono::duration_cast<std::chrono::minutes>(ns.time_since_epoch());
        ns -= m;
        auto s = std::chrono::duration_cast<std::chrono::seconds>(ns.time_since_epoch());
        ns -= s;
        ss << std::setw(2) << h.count() << ':'
            << std::setw(2) << m.count() << ':'
            << std::setw(2) << s.count() << '.'
            << std::setw(4) << ns.time_since_epoch().count();
        return ss.str();
    }

}

namespace std
{

std::ostream& operator <<(std::ostream& ss, std::chrono::high_resolution_clock::time_point ns) 
{
    typedef std::chrono::duration<int, std::ratio<86400>> days;
    ss.fill('0');
    auto d = std::chrono::duration_cast<days>(ns.time_since_epoch());
    ns -= d;
    auto h = std::chrono::duration_cast<std::chrono::hours>(ns.time_since_epoch());
    ns -= h;
    auto m = std::chrono::duration_cast<std::chrono::minutes>(ns.time_since_epoch());
    ns -= m;
    auto s = std::chrono::duration_cast<std::chrono::seconds>(ns.time_since_epoch());
    ns -= s;
    ss << std::setw(2) << h.count() << ':'
        << std::setw(2) << m.count() << ':'
        << std::setw(2) << s.count() << '.'
        << std::setw(4) << ns.time_since_epoch().count();
    ss << ns.time_since_epoch().count() << " ns";
    return ss;
}
std::ostream& operator <<(std::ostream& ss, std::chrono::milliseconds ns){
    typedef std::chrono::duration<int, std::ratio<86400>> days;
    ss.fill('0');
    auto d = std::chrono::duration_cast<days>(ns);
    ns -= d;
    auto h = std::chrono::duration_cast<std::chrono::hours>(ns);
    ns -= h;
    auto m = std::chrono::duration_cast<std::chrono::minutes>(ns);
    ns -= m;
    auto s = std::chrono::duration_cast<std::chrono::seconds>(ns);
    ns -= s;
    ss << std::setw(2) << h.count() << ':'
        << std::setw(2) << m.count() << ':'
        << std::setw(2) << s.count() << '.'
        << std::setw(4) << ns.count();
    ss << ns.count() << " ns";
    return ss;
}

std::ostream& operator <<(std::ostream& ss, std::chrono::microseconds ns){
    typedef std::chrono::duration<int, std::ratio<86400>> days;
    ss.fill('0');
    auto d = std::chrono::duration_cast<days>(ns);
    ns -= d;
    auto h = std::chrono::duration_cast<std::chrono::hours>(ns);
    ns -= h;
    auto m = std::chrono::duration_cast<std::chrono::minutes>(ns);
    ns -= m;
    auto s = std::chrono::duration_cast<std::chrono::seconds>(ns);
    ns -= s;
    ss << std::setw(2) << h.count() << ':'
        << std::setw(2) << m.count() << ':'
        << std::setw(2) << s.count() << '.'
        << std::setw(4) << ns.count();
    ss << ns.count() << " us";
    return ss;
}

std::ostream& operator <<(std::ostream& ss, std::chrono::nanoseconds ns){
    typedef std::chrono::duration<int, std::ratio<86400>> days;
    ss.fill('0');
    auto d = std::chrono::duration_cast<days>(ns);
    ns -= d;
    auto h = std::chrono::duration_cast<std::chrono::hours>(ns);
    ns -= h;
    auto m = std::chrono::duration_cast<std::chrono::minutes>(ns);
    ns -= m;
    auto s = std::chrono::duration_cast<std::chrono::seconds>(ns);
    ns -= s;
    ss << std::setw(2) << h.count() << ':'
        << std::setw(2) << m.count() << ':'
        << std::setw(2) << s.count() << '.'
        << std::setw(4) << ns.count();
    ss << ns.count() << " ns";
    return ss;
}

std::ostream& operator <<(std::ostream& ss, std::chrono::seconds ns){
    typedef std::chrono::duration<int, std::ratio<86400>> days;
    ss.fill('0');
    auto d = std::chrono::duration_cast<days>(ns);
    ns -= d;
    auto h = std::chrono::duration_cast<std::chrono::hours>(ns);
    ns -= h;
    auto m = std::chrono::duration_cast<std::chrono::minutes>(ns);
    ns -= m;
    auto s = std::chrono::duration_cast<std::chrono::seconds>(ns);
    ns -= s;
    ss << std::setw(2) << h.count() << ':'
        << std::setw(2) << m.count() << ':'
        << std::setw(2) << s.count() << '.'
        << std::setw(4) << ns.count();
    ss << ns.count() << " s";
    return ss;
}
}
