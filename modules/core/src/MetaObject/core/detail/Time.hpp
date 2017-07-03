#pragma once
#include <MetaObject/detail/Export.hpp>
#include <chrono>
#include <boost/optional.hpp>
#include <boost/optional/optional_io.hpp>
namespace mo {
typedef std::chrono::system_clock::time_point Time_t;
typedef boost::optional<Time_t> OptionalTime_t;

template<class T> struct TimePrefix{
    static Time_t convert(unsigned long val){
        return Time_t(T(val));
    }
};

static const auto ms = TimePrefix<std::chrono::milliseconds>();
static const auto ns = TimePrefix<std::chrono::nanoseconds>();
static const auto us = TimePrefix<std::chrono::microseconds>();
static const auto second = TimePrefix<std::chrono::seconds>();

template<class T>
Time_t operator*(const TimePrefix<T>& lhs, double rhs){
    (void)lhs;
    return TimePrefix<T>::convert(rhs);
}
template<class T>
Time_t operator*(double rhs, const TimePrefix<T>& lhs){
    (void)lhs;
    return TimePrefix<T>::convert(rhs);
}
std::ostream& operator <<(std::ostream& lhs, const mo::Time_t& rhs);

typedef mo::Time_t(*GetTime_f)();
MO_EXPORTS mo::Time_t getCurrentTime();
MO_EXPORTS void setTimeSource(GetTime_f timefunc);
} // namespace mo
namespace std{
std::ostream& operator <<(std::ostream& lhs, const std::chrono::system_clock::time_point& rhs);
std::ostream& operator <<(std::ostream& lhs, const std::chrono::milliseconds& rhs);
std::ostream& operator <<(std::ostream& lhs, const std::chrono::microseconds& rhs);
std::ostream& operator <<(std::ostream& lhs, const std::chrono::nanoseconds& rhs);
std::ostream& operator <<(std::ostream& lhs, const std::chrono::seconds& rhs);
}
namespace cereal {
template <class AR>
double save_minimal(const AR& ar, const mo::Time_t& time) {
    (void)ar;
    return time.time_since_epoch().count();
}
template <class AR>
void load_minimal(AR& ar, mo::Time_t& time, const double& value) {
    (void)ar;
    time = mo::ns * value;
}
}
