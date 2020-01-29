#ifndef MO_CORE_LOGGING_HPP
#define MO_CORE_LOGGING_HPP

#include <MetaObject/detail/Export.hpp>
#if !(defined(__GNUC__) && __GNUC__ == 4 && defined(__NVCC__))
#include <spdlog/fmt/fmt.h>
#include <spdlog/spdlog.h>
#endif
#include "logging_macros.hpp"
#include <sstream>

struct SystemTable;
namespace mo
{
    MO_EXPORTS void initLogging();

    MO_EXPORTS spdlog::details::registry& getLoggerRegistry();
    MO_EXPORTS spdlog::details::registry& getLoggerRegistry(SystemTable& table);
    MO_EXPORTS spdlog::logger& getDefaultLogger();
    MO_EXPORTS std::shared_ptr<spdlog::logger> getLogger();

} // namespace mo
#endif // MO_CORE_LOGGING_HPP
