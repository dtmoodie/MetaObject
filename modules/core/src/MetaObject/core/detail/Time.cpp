#include "Time.hpp"
#include <chrono>
#include <iomanip>
#include <ostream>
#include <sstream>

namespace mo
{
    static Time::GetTime_f time_source = nullptr;

    Time Time::now()
    {
        if (time_source)
        {
            return time_source();
        }
        return std::chrono::high_resolution_clock::now();
    }

    void Time::setTimeSource(Time::GetTime_f timefunc)
    {
        time_source = timefunc;
    }

    Time::Time(const std::chrono::high_resolution_clock::time_point& t)
        : std::chrono::high_resolution_clock::time_point(t)
    {
    }

    Time::Time(const Duration& d)
        : std::chrono::high_resolution_clock::time_point(d)
    {
    }

    std::string Time::print() const
    {
        std::stringstream ss;
        print(ss);
        return ss.str();
    }

    using Days = std::chrono::duration<int, std::ratio<86400>>;
    void Time::print(std::ostream& ss,
                     const bool print_days,
                     const bool hours,
                     const bool minutes,
                     const bool seconds,
                     const bool nanoseconds) const
    {

        ss.fill('0');
        auto ns = *this;
        auto d = std::chrono::duration_cast<Days>(ns.time_since_epoch());
        ns -= d;
        auto h = std::chrono::duration_cast<std::chrono::hours>(ns.time_since_epoch());
        ns -= h;
        auto m = std::chrono::duration_cast<std::chrono::minutes>(ns.time_since_epoch());
        ns -= m;
        auto s = std::chrono::duration_cast<std::chrono::seconds>(ns.time_since_epoch());
        ns -= s;
        if (print_days)
        {
            ss << std::setw(2) << d.count();
            ss << ':';
        }
        if (hours)
        {
            if (print_days)
            {
                ss << ':';
            }
            ss << std::setw(2) << h.count();
        }
        if (minutes)
        {
            if (hours || print_days)
            {
                ss << ':';
            }
            ss << std::setw(2) << m.count();
        }
        if (seconds)
        {
            if (minutes)
            {
                ss << ':';
            }

            ss << std::setw(2) << s.count();
        }
        if (nanoseconds)
        {
            if (seconds)
            {
                ss << '.';
            }
            ss << std::setw(4) << ns.time_since_epoch().count();
        }
    }

    double Time::seconds() const
    {
        auto ns = *this;
        auto d = std::chrono::duration_cast<Days>(ns.time_since_epoch());
        ns -= d;
        auto h = std::chrono::duration_cast<std::chrono::hours>(ns.time_since_epoch());
        ns -= h;
        auto m = std::chrono::duration_cast<std::chrono::minutes>(ns.time_since_epoch());
        ns -= m;
        auto s = std::chrono::duration_cast<std::chrono::seconds>(ns.time_since_epoch());
        ns -= s;
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(ns.time_since_epoch());
        return s.count() + (ms.count() / 1000.0);
    }

    FrameNumber::FrameNumber(const uint64_t v)
        : val(v)
    {
    }

    uint64_t FrameNumber::max()
    {
        return std::numeric_limits<uint64_t>::max();
    }

    bool FrameNumber::valid() const
    {
        return val != max();
    }

    FrameNumber::operator uint64_t() const
    {
        return val;
    }

    FrameNumber& FrameNumber::operator=(const uint32_t v)
    {
        val = v;
        return *this;
    }

    bool FrameNumber::operator==(const FrameNumber& v) const
    {
        return val == v.val;
    }
}

namespace std
{
    std::ostream& operator<<(std::ostream& lhs, const mo::Time& rhs)
    {
        rhs.print(lhs);
        return lhs;
    }
}
