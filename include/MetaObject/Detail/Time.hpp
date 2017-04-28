#pragma once
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
}