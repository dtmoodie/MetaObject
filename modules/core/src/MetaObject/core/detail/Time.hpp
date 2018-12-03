#pragma once
#include <MetaObject/detail/Export.hpp>
#include <boost/optional.hpp>
#include <boost/optional/optional_io.hpp>

#include <ct/reflect.hpp>

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
        operator uint64_t&();
        operator uint64_t() const;
        FrameNumber& operator=(const uint64_t v);
        bool operator==(const FrameNumber&) const;

        uint64_t val = max();
    };

    using OptionalTime = boost::optional<Time>;

    template <class T>
    struct MorePreciseTime;

    template <>
    struct MorePreciseTime<std::chrono::hours>
    {
        using type = std::chrono::minutes;
    };
    template <>
    struct MorePreciseTime<std::chrono::minutes>
    {
        using type = std::chrono::seconds;
    };
    template <>
    struct MorePreciseTime<std::chrono::seconds>
    {
        using type = std::chrono::milliseconds;
    };
    template <>
    struct MorePreciseTime<std::chrono::milliseconds>
    {
        using type = std::chrono::microseconds;
    };
    template <>
    struct MorePreciseTime<std::chrono::microseconds>
    {
        using type = std::chrono::nanoseconds;
    };

    template <class T>
    struct TimePrefix
    {
        static Duration convert(const int64_t val)
        {
            return Duration(T(val));
        }

        static Duration convert(double val)
        {
            using MorePrecise = typename MorePreciseTime<T>::type;
            const unsigned long integral = static_cast<unsigned long>(val);
            T whole(integral);
            val -= integral;
            using Ratio = std::ratio_divide<typename MorePrecise::period, typename T::period>;
            const auto num = Ratio::num;
            const auto den = Ratio::den;
            auto fr = static_cast<unsigned long>(val * den / num);
            MorePrecise fraction(fr);
            return Duration(whole + fraction);
        }
    };

    static const auto ns = TimePrefix<std::chrono::nanoseconds>();
    static const auto us = TimePrefix<std::chrono::microseconds>();
    static const auto ms = TimePrefix<std::chrono::milliseconds>();
    static const auto second = TimePrefix<std::chrono::seconds>();
    static const auto minutes = TimePrefix<std::chrono::minutes>();
    static const auto hours = TimePrefix<std::chrono::hours>();

    template <class T>
    Duration operator*(const TimePrefix<T>& /*lhs*/, const double rhs)
    {
        return TimePrefix<T>::convert(rhs);
    }

    template <class T>
    Duration operator*(const double rhs, const TimePrefix<T>& /*lhs*/)
    {
        return TimePrefix<T>::convert(rhs);
    }

    template <class T>
    Duration operator*(const TimePrefix<T>& /*lhs*/, const int64_t rhs)
    {
        return TimePrefix<T>::convert(rhs);
    }

    template <class T>
    Duration operator*(const int64_t rhs, const TimePrefix<T>& /*lhs*/)
    {
        return TimePrefix<T>::convert(rhs);
    }

    template <class T>
    Duration operator*(const TimePrefix<T>& /*lhs*/, const int32_t rhs)
    {
        return TimePrefix<T>::convert(static_cast<int64_t>(rhs));
    }

    template <class T>
    Duration operator*(const int32_t rhs, const TimePrefix<T>& /*lhs*/)
    {
        return TimePrefix<T>::convert(static_cast<int64_t>(rhs));
    }
} // namespace mo

namespace std
{
    MO_EXPORTS std::ostream& operator<<(std::ostream& lhs, const mo::Time& rhs);
    namespace chrono
    {
        template <class Rep,
                  class Period,
                  class = typename std::enable_if<std::chrono::duration<Rep, Period>::min() <
                                                  std::chrono::duration<Rep, Period>::zero()>::type>
        constexpr std::chrono::duration<Rep, Period> abs(const duration<Rep, Period> d)
        {
            return d >= d.zero() ? d : -d;
        }
    }
}

namespace cereal
{
    template <class AR>
    double save_minimal(const AR& /*ar*/, const mo::Time& time)
    {
        return time.seconds();
    }

    template <class AR>
    void load_minimal(AR& /*ar*/, mo::Time& time, const double& value)
    {
        time = mo::Time(value * mo::second);
    }
}

namespace ct
{
    REFLECT_BEGIN(mo::FrameNumber)
        PUBLIC_ACCESS(val)
        MEMBER_FUNCTION("valid", &mo::FrameNumber::valid)
    REFLECT_END;
}
