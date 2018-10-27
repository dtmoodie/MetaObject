#pragma once
#include <MetaObject/detail/Export.hpp>
#include <boost/optional.hpp>
#include <boost/optional/optional_io.hpp>
#include <chrono>
namespace mo
{
    using Duration = std::chrono::high_resolution_clock::duration;

    struct Time : public std::chrono::high_resolution_clock::time_point
    {
        using GetTime_f = mo::Time (*)();

        static Time now();
        static void setTimeSource(GetTime_f timefunc);

        Time() = default;
        Time(const std::chrono::high_resolution_clock::time_point& t);
        Time(const Duration& d);

        double seconds() const;

        std::string print() const;
        void print(std::ostream& os,
                   const bool print_days = false,
                   const bool print_hours = false,
                   const bool print_minutes = true,
                   const bool print_seconds = true,
                   const bool print_nanoseconds = false) const;
    };

    struct FrameNumber
    {
        static uint64_t max();

        FrameNumber(const uint64_t v = max());

        bool valid() const;
        operator uint64_t() const;
        FrameNumber& operator=(const uint64_t v);
        bool operator==(const FrameNumber&) const;

        uint64_t val = max();
    };

    using OptionalTime = boost::optional<Time>;

    template <class T>
    struct TimePrefix
    {
        static Duration convert(unsigned long val)
        {
            return Duration(T(val));
        }
    };

    static const auto ms = TimePrefix<std::chrono::milliseconds>();
    static const auto ns = TimePrefix<std::chrono::nanoseconds>();
    static const auto us = TimePrefix<std::chrono::microseconds>();
    static const auto second = TimePrefix<std::chrono::seconds>();

    template <class T>
    Duration operator*(const TimePrefix<T>& /*lhs*/, double rhs)
    {
        return TimePrefix<T>::convert(static_cast<unsigned long>(rhs));
    }

    template <class T>
    Duration operator*(double rhs, const TimePrefix<T>& /*lhs*/)
    {
        return TimePrefix<T>::convert(rhs);
    }
} // namespace mo

namespace std
{
    MO_EXPORTS std::ostream& operator<<(std::ostream& lhs, const mo::Time& rhs);
}

namespace cereal
{
    template <class AR>
    double save_minimal(const AR& /*ar*/, const mo::Time& time)
    {
        return time.time_since_epoch().count();
    }

    template <class AR>
    void load_minimal(AR& /*ar*/, mo::Time& time, const double& value)
    {
        time = mo::Time(mo::ns * value);
    }
}
