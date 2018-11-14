#pragma once

#include <MetaObject/detail/Export.hpp>
#include <boost/stacktrace.hpp>
#include <spdlog/spdlog.h>

#include "logging_macros.hpp"

#include <sstream>
namespace spdlog
{
    namespace details
    {
        class registry;
    }
}
namespace mo
{
    MO_EXPORTS void initLogging();

    MO_EXPORTS spdlog::details::registry& getLogerRegistry();
    MO_EXPORTS spdlog::logger& getDefaultLogger();

} // namespace mo
