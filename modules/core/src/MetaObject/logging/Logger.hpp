#ifndef MO_LOGGING_LOGGER_HPP
#define MO_LOGGING_LOGGER_HPP
#include <MetaObject/detail/Export.hpp>
#include <boost/stacktrace.hpp>
#include <spdlog/logger.h>
namespace mo
{
    /*struct MO_EXPORTS Logger
    {
        Logger(std::shared_ptr<spdlog::logger>&& logger)
            : m_logger(std::move(logger))
        {
        }

        template <typename... Args>
        void log(level::level_enum lvl, const char* fmt, const Args&... args)
        {
            m_logger->log(lvl, fmt, args...);
        }

        template <typename... Args>
        void log(level::level_enum lvl, const char* msg)
        {
            m_logger->log(lvl, msg);
        }

        template <typename Arg1, typename... Args>
        void trace(const char* fmt, const Arg1& arg, const Args&... args)
        {
            m_logger->log(fmt, arg, args...);
        }

        template <typename Arg1, typename... Args>
        void debug(const char* fmt, const Arg1& arg, const Args&... args)
        {
            m_logger->debug(fmt, arg, args...);
        }

        template <typename Arg1, typename... Args>
        void info(const char* fmt, const Arg1& arg, const Args&... args)
        {
            m_logger->info(fmt, arg, args...);
        }

        template <typename Arg1, typename... Args>
        void warn(const char* fmt, const Arg1& arg, const Args&... args)
        {
            m_logger->warn(fmt, arg, args...);
        }

        template <typename Arg1, typename... Args>
        void error(const char* fmt, const Arg1& arg, const Args&... args)
        {
            m_logger->error(fmt, arg, args...);
        }

        template <typename Arg1, typename... Args>
        void critical(const char* fmt, const Arg1& arg, const Args&... args)
        {
            m_logger->critical(fmt, arg, args...);
        }

        template <typename T>
        void log(level::level_enum lvl, const T& arg)
        {
            m_logger->log(lvl, arg);
        }

        template <typename T>
        void trace(const T& msg)
        {
            m_logger->trace(msg);
        }

        template <typename T>
        void debug(const T& msg)
        {
            m_logger->debug(msg);
        }

        template <typename T>
        void info(const T& msg)
        {
            m_logger->info(msg);
        }

        template <typename T>
        void warn(const T& msg)
        {
            m_logger->warn(msg);
        }

        template <typename T>
        void error(const T& msg)
        {
            m_logger->error(msg);
        }

        template <typename T>
        void critical(const T& msg)
        {
            m_logger->critical(msg);
        }

        bool should_log(level::level_enum msg_level) const
        {
            return m_logger->should_log(msg_level);
        }
        void set_level(level::level_enum log_level)
        {
            m_logger->set_level(log_level);
        }
        level::level_enum level() const
        {
            return m_logger->level();
        }
        const std::string& name() const
        {
            return m_logger->name();
        }
        void set_pattern(const std::string& pattern, pattern_time_type pattern_time = pattern_time_type::local)
        {
            m_logger->set_pattern(pattern, pattern_time);
        }
        void set_formatter(formatter_ptr msg_formatter)
        {
            m_logger->set_formatter(msg_formatter);
        }

        // automatically call flush() if message level >= log_level
        void flush_on(level::level_enum log_level)
        {
            m_logger->flush_on(log_level);
        }

        void flush()
        {
            m_logger->flush();
        }

        const std::vector<sink_ptr>& sinks() const
        {
            return m_logger->sinks();
        }

        // error handler
        void set_error_handler(log_err_handler err_handler)
        {
            m_logger->set_error_handler(err_handler);
        }

        log_err_handler error_handler()
        {
            return m_logger->error_handler();
        }

        template <class T, class U>
        void assertEqual(const T& lhs, const U& rhs)
        {
            if (lhs == rhs)
            {
                boost::stacktrace::callstack cs;
                std::string msg = fmt::format("lhs ({}) != rhs ({})\n{}", lhs, rhs, cs);
                error(msg);
                throw std::runtime_error(std::move(msg));
            }
        }

        template <class T, class U>
        void assertNotEqual(const T& lhs, const U& rhs)
        {
            if (lhs != rhs)
            {
                boost::stacktrace::callstack cs;
                std::string msg = fmt::format("lhs ({}) == rhs ({})\n{}", lhs, rhs, cs);
                error(msg);
                throw std::runtime_error(std::move(msg));
            }
        }

        template <class T>
        void assertNotNull(const T& ptr)
        {
            assertNotEqual(ptr, nullptr);
        }

      private:
        std::shared_ptr<Logger> m_logger;
    };*/
} // namespace mo
#endif // MO_LOGGING_LOGGER_HPP
