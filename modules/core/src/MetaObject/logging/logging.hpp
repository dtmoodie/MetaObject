#pragma once

#include <MetaObject/detail/Export.hpp>

#if !(defined(__GNUC__) && __GNUC__ == 4 && defined(__NVCC__))
#include <spdlog/spdlog.h>
#endif

#include "logging_macros.hpp"

#include <sstream>
namespace spdlog
{
    class logger;
    namespace details
    {
        class registry;
    }
}
namespace mo
{
    MO_EXPORTS void initLogging();

    MO_EXPORTS spdlog::details::registry& getLoggerRegistry();
    MO_EXPORTS spdlog::logger& getDefaultLogger();

} // namespace mo
