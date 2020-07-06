#ifndef MO_LOGGING_CALLSTACK_HPP
#define MO_LOGGING_CALLSTACK_HPP
#include <MetaObject/detail/Export.hpp>

#include <boost/stacktrace.hpp>

namespace mo
{

    struct MO_EXPORTS IExceptionWithCallstack
    {
        virtual ~IExceptionWithCallstack();

        virtual const boost::stacktrace::stacktrace& getCallstack() const = 0;
    };

    template <class T>
    struct TExceptionWithCallstack : IExceptionWithCallstack
    {
        TExceptionWithCallstack(T&& msg)
            : m_msg(std::move(msg))
        {
        }

        const boost::stacktrace::stacktrace& getCallstack() const override
        {
            return m_callstack;
        }

        const T& getException() const
        {
            return m_msg;
        }

      private:
        T m_msg;
        boost::stacktrace::stacktrace m_callstack;
    };

    template <class T>
    void throwWithCallstack(T&& msg)
    {
        throw TExceptionWithCallstack<T>(std::forward<T>(msg));
    }
} // namespace mo

#endif // MO_LOGGING_CALLSTACK_HPP
