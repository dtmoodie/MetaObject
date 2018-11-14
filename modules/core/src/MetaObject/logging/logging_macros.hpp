#pragma once

#define LOG_EVERY_N_VARNAME(base, line) LOG_EVERY_N_VARNAME_CONCAT(base, line)
#define LOG_EVERY_N_VARNAME_CONCAT(base, line) base##line

#define LOG_OCCURRENCES LOG_EVERY_N_VARNAME(occurrences_, __LINE__)
#define LOG_OCCURRENCES_MOD_N LOG_EVERY_N_VARNAME(occurrences_mod_n_, __LINE__)

#define LOG_THEN_THROW(level, ...)                                                                                     \
    do                                                                                                                 \
    {                                                                                                                  \
        const std::string msg = fmt::format(__VA_ARGS__);                                                              \
        mo::getDefaultLogger().level(msg);                                                                             \
        throw std::runtime_error(msg);                                                                                 \
    } while (0)

#define THROW(level, ...) LOG_THEN_THROW(level, __VA_ARGS__)

#define MO_ASSERT(CHECK)                                                                                               \
    if (!(CHECK))                                                                                                      \
    THROW(error, #CHECK)

#define MO_ASSERT_FMT(CHECK, ...)                                                                                      \
    if (!(CHECK))                                                                                                      \
    THROW(error, __VA_ARGS__)

#define MO_LOG(LEVEL, ...) getDefaultLogger().LEVEL(__VA_ARGS__)
