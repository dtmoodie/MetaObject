#pragma once
#include <MetaObject/detail/Export.hpp>
#include <boost/version.hpp>
#include <boost/units/systems/si.hpp>
#include <boost/units/systems/si/prefixes.hpp>
#include <boost/units/systems/si/time.hpp>
#include <boost/units/io.hpp>
#include <boost/optional.hpp>
#include <boost/optional/optional_io.hpp>

namespace mo {
static const auto milli = boost::units::si::milli;
static const auto nano = boost::units::si::nano;
static const auto micro = boost::units::si::micro;
static const auto second = boost::units::si::second;
static const auto millisecond = milli * second;
static const auto nanosecond = nano * second;
static const auto microseconds = micro * second;
static const auto ms = millisecond;
static const auto ns = nanosecond;
static const auto us = microseconds;
typedef boost::units::quantity<boost::units::si::time> Time_t;
typedef boost::optional<Time_t> OptionalTime_t;
typedef mo::Time_t(*GetTime_f)();
MO_EXPORTS mo::Time_t getCurrentTime();
MO_EXPORTS void setTimeSource(GetTime_f timefunc);
}

namespace cereal {
template <class AR>
double save_minimal(const AR& ar, const mo::Time_t& time) {
    (void)ar;
    return time.value();
}
template <class AR>
void load_minimal(AR& ar, mo::Time_t& time, const double& value) {
    (void)ar;
    time.from_value(value);
}
}
