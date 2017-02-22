#pragma once
#include "IParameter.hpp"
#include <boost/units/systems/si.hpp>
#include <boost/units/systems/si/prefixes.hpp>
#include <boost/units/systems/si/time.hpp>
#include <boost/units/io.hpp>

namespace mo
{

class ICoordinateSystem;
typedef boost::units::quantity<boost::units::si::time> time_t;
static const auto milli = boost::units::si::milli;
static const auto nano = boost::units::si::nano;
static const auto second = boost::units::si::second;
static const auto millisecond = milli * second;
static const auto nanosecond = nano * second;
static const auto ms = millisecond;
static const auto ns = nanosecond;

class MO_EXPORTS StampedParameter: virtual public IParameter
{
public:
    time_t             GetTimestamp() const;
    template<class T>
    IParameter* SetTimestamp(T&& ts)
    {
        _timestamp = time_t(ts);
        return this;
    }
    IParameter* SetFrameNumber(size_t fn);
    size_t      GetFrameNumber() const;

    void SetCoordinateSystem(ICoordinateSystem* system);
    ICoordinateSystem* GetCoordinateSystem() const;
};

} // namespace mo
