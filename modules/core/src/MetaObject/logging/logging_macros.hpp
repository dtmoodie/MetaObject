#ifndef MO_LOGGING_LOGGING_MACROS_HPP
#define MO_LOGGING_LOGGING_MACROS_HPP
#include "callstack.hpp"
#include <spdlog/fmt/fmt.h>
#include <spdlog/fmt/ostr.h>

#if defined(__GNUC__) && __GNUC__ == 4 && defined(__NVCC__)

// These empy definitions are for GCC 4.8 with cuda 6.5 since it can't handle fmt
#define LOG_EVERY_N_VARNAME(base, line)
#define LOG_EVERY_N_VARNAME_CONCAT(base, line)

#define LOG_OCCURRENCES
#define LOG_OCCURRENCES_MOD_N

#define LOG_THEN_THROW(level, ...)

#define THROW(level, ...)

#define MO_ASSERT(CHECK)

#define MO_ASSERT_GT(LHS, RHS)

#define MO_ASSERT_GE(LHS, RHS)

#define MO_ASSERT_FMT(CHECK, ...)

#define MO_LOG(LEVEL, ...)

#else
// All other competent compilers
#define LOG_EVERY_N_VARNAME(base, line) LOG_EVERY_N_VARNAME_CONCAT(base, line)
#define LOG_EVERY_N_VARNAME_CONCAT(base, line) base##line

#define LOG_OCCURRENCES LOG_EVERY_N_VARNAME(occurrences_, __LINE__)
#define LOG_OCCURRENCES_MOD_N LOG_EVERY_N_VARNAME(occurrences_mod_n_, __LINE__)

#define LOG_THEN_THROW(level, ...)                                                                                     \
    do                                                                                                                 \
    {                                                                                                                  \
        const std::string msg = fmt::format(__VA_ARGS__);                                                              \
        mo::getDefaultLogger().level(msg);                                                                             \
        mo::throwWithCallstack(std::runtime_error(msg));                                                               \
    } while (0)

#define THROW(level, ...) LOG_THEN_THROW(level, __VA_ARGS__)

#define MO_ASSERT(CHECK)                                                                                               \
    if (!(CHECK))                                                                                                      \
    THROW(error, #CHECK)

#define MO_ASSERT_EQ(LHS, RHS)                                                                                         \
    if ((LHS) != (RHS))                                                                                                \
    THROW(error, #LHS " != " #RHS " [{} != {}]", LHS, RHS)

#define MO_ASSERT_FMT(CHECK, ...)                                                                                      \
    if (!(CHECK))                                                                                                      \
    THROW(error, __VA_ARGS__)

#define MO_ASSERT_GT(LHS, RHS)                                                                                         \
    if (!((LHS) > (RHS)))                                                                                              \
    THROW(error, #LHS " <= " #RHS " [{} <= {}]", LHS, RHS)

#define MO_ASSERT_GE(LHS, RHS)                                                                                         \
    if (!((LHS) >= (RHS)))                                                                                             \
    THROW(error, #LHS " < " #RHS " [{} < {}]", LHS, RHS)

#define MO_LOG(LEVEL, ...) mo::getDefaultLogger().LEVEL(__VA_ARGS__)
#endif

#endif // MO_LOGGING_LOGGING_MACROS_HPP
