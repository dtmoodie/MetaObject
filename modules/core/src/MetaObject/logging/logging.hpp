#pragma once
#include "MetaObject/detail/Export.hpp"
#include "MetaObject/logging/logging_macros.hpp"
#include <boost/log/trivial.hpp>
#include <functional>

#ifndef WIN32
#include "RuntimeObjectSystem/RuntimeLinkLibrary.h"
RUNTIME_COMPILER_LINKLIBRARY("-lboost_log")
RUNTIME_COMPILER_LINKLIBRARY("-lboost_log_setup")
#endif

#include <sstream>
// https://github.com/Microsoft/CNTK/blob/7c811de9e33d0184fdf340cd79f4f17faacf41cc/Source/Common/Include/ExceptionWithCallStack.h
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings
#endif

#ifdef _WIN32
#define NOMINMAX
#pragma comment(lib, "Dbghelp.lib")
#else
#include <cxxabi.h>
#include <execinfo.h>
#endif

#ifdef MO_LOG
#undef MO_LOG
#endif
#ifdef MO_CHECK_EQ
#undef MO_CHECK_EQ
#endif
#ifdef MO_CHECK_NE
#undef MO_CHECK_NE
#endif
#ifdef MO_CHECK_LE
#undef MO_CHECK_LE
#endif
#ifdef MO_CHECK_LT
#undef MO_CHECK_LT
#endif
#ifdef MO_CHECK_GE
#undef MO_CHECK_GE
#endif
#ifdef MO_CHECK_GT
#undef MO_CHECK_GT
#endif
#ifdef MO_CHECK_OP
#undef MO_CHECK_OP
#endif
#ifdef MO_ASSERT_OP
#undef MO_ASSERT_OP
#endif
#ifdef MO_LOG_FIRST_N
#undef MO_LOG_FIRST_N
#endif
#ifdef MO_CHECK_OP
#undef MO_CHECK_OP
#endif
#ifdef MO_CHECK_GT
#undef MO_CHECK_GT
#endif
#ifdef MO_CHECK_GE
#undef MO_CHECK_GE
#endif
#ifdef MO_CHECK_LT
#undef MO_CHECK_LT
#endif
#ifdef MO_CHECK_LE
#undef MO_CHECK_LE
#endif
#ifdef MO_CHECK_NE
#undef MO_CHECK_NE
#endif
#ifdef MO_DISCARD_MESSAGE
#undef MO_DISCARD_MESSAGE
#endif
#ifdef ALIAS_BOOST_LOG_SEVERITIES
namespace boost {
namespace log {
    inline namespace BOOST_LOG_VERSION_NAMESPACE {
        namespace trivial {
            constexpr severity_level FATAL   = fatal;
            constexpr severity_level ERROR   = error;
            constexpr severity_level WARNING = warning;
            constexpr severity_level INFO    = info;
        }
    }
}
}
#endif
#define MO_DISCARD_MESSAGE true ? (void)0 : mo::LogMessageVoidify() & mo::eat_message().stream()

#define MO_LOG_EVERY_N(severity, n)                              \
    static int LOG_OCCURRENCES = 0, LOG_OCCURRENCES_MOD_N = 0;   \
    ++LOG_OCCURRENCES;                                           \
    if (++LOG_OCCURRENCES_MOD_N > n) LOG_OCCURRENCES_MOD_N -= n; \
    if (LOG_OCCURRENCES_MOD_N == 1)                              \
    MO_LOG(severity)

#define MO_LOG_FIRST_N(severity, n)              \
    static int LOG_OCCURRENCES = 0;              \
    if (LOG_OCCURRENCES <= n) ++LOG_OCCURRENCES; \
    if (LOG_OCCURRENCES <= n) MO_LOG(severity)

#define MO_OBJ_LOG(severity) BOOST_LOG_TRIVIAL(severity) << "[" << this->GetTypeName() << "::" __FUNCTION__ "] "

#define MO_LOG(severity) BOOST_LOG_TRIVIAL(severity) << "[" << __FUNCTION__ << "] "

#define DOIF(condition, expression, severity)                                          \
    if (condition) {                                                                   \
        MO_LOG(severity) << #condition << " is true, thus performing " << #expression; \
        expression;                                                                    \
    } else {                                                                           \
        MO_LOG(severity) << #condition << " failed";                                   \
    }

#define DOIF_LOG_PASS(condition, expression, severity)                                 \
    if (condition) {                                                                   \
        MO_LOG(severity) << #condition << " is true, thus performing " << #expression; \
        expression;                                                                    \
    }
#define DOIF_LOG_FAIL(condition, expression, severity)                                           \
    if (condition) {                                                                             \
        expression;                                                                              \
    } else {                                                                                     \
        MO_LOG(severity) << "Unable to perform " #expression " due to " #condition << " failed"; \
    }

#define LOGIF_EQ(lhs, rhs, severity)                                           \
    if (lhs == rhs) MO_LOG(severity) << "if(" << #lhs << " == " << #rhs << ")" \
                                     << "[" << lhs << " == " << rhs << "]";
#define LOGIF_NEQ(lhs, rhs, severity)                                          \
    if (lhs != rhs) MO_LOG(severity) << "if(" << #lhs << " != " << #rhs << ")" \
                                     << "[" << lhs << " != " << rhs << "]";

#define MO_ASSERT_OP(op, lhs, rhs) \
    if (!(lhs op rhs)) mo::ThrowOnDestroy(__FUNCTION__, __FILE__, __LINE__).stream() << "[" << #lhs << " " << #op << " " << #rhs << "] - Failed (" << lhs << " " << #op << " " << rhs << ")"

#define MO_ASSERT_EQ(lhs, rhs) MO_ASSERT_OP(==, lhs, rhs)
#define MO_ASSERT_NE(lhs, rhs) MO_ASSERT_OP(!=, lhs, rhs)
#define MO_ASSERT_LE(lhs, rhs) MO_ASSERT_OP(<=, lhs, rhs)
#define MO_ASSERT_LT(lhs, rhs) MO_ASSERT_OP(<, lhs, rhs)
#define MO_ASSERT_GE(lhs, rhs) MO_ASSERT_OP(>=, lhs, rhs)
#define MO_ASSERT_GT(lhs, rhs) MO_ASSERT_OP(>, lhs, rhs)
#define MO_ASSERT(exp) \
    if (!(exp)) mo::ThrowOnDestroy(__FUNCTION__, __FILE__, __LINE__).stream() << "[" << #exp << "] FAILED! "

#define CHECK_OP(op, lhs, rhs, severity) \
    if (lhs op rhs) MO_LOG(severity) << "[" << #lhs << " " << #op << " " << #rhs << "] - Failed (" << lhs << " " << #op << " " << rhs << ")"

#define MO_CHECK_EQ(lhs, rhs) MO_ASSERT_EQ(lhs, rhs)
#define MO_CHECK_NE(lhs, rhs) MO_ASSERT_NE(lhs, rhs)
#define MO_CHECK_LE(lhs, rhs) MO_ASSERT_LE(lhs, rhs)
#define MO_CHECK_LT(lhs, rhs) MO_ASSERT_LT(lhs, rhs)
#define MO_CHECK_GE(lhs, rhs) MO_ASSERT_GE(lhs, rhs)
#define MO_CHECK_GT(lhs, rhs) MO_ASSERT_GT(lhs, rhs)

#ifdef _DEBUG
#define DBG_CHECK_EQ(lhs, rhs, severity) CHECK_OP(==, lhs, rhs, severity)
#define DBG_CHECK_NE(lhs, rhs, severity) CHECK_OP(!=, lhs, rhs, severity)
#define DBG_CHECK_LE(lhs, rhs, severity) CHECK_OP(<=, lhs, rhs, severity)
#define DBG_CHECK_LT(lhs, rhs, severity) CHECK_OP(<, lhs, rhs, severity)
#define DBG_CHECK_GE(lhs, rhs, severity) CHECK_OP(>=, lhs, rhs, severity)
#define DBG_CHECK_GT(lhs, rhs, severity) CHECK_OP(>, lhs, rhs, severity)
#else
#define DBG_CHECK_EQ(lhs, rhs, severity) DISCARD_MESSAGE
#define DBG_CHECK_NE(lhs, rhs, severity) DISCARD_MESSAGE
#define DBG_CHECK_LE(lhs, rhs, severity) DISCARD_MESSAGE
#define DBG_CHECK_LT(lhs, rhs, severity) DISCARD_MESSAGE
#define DBG_CHECK_GE(lhs, rhs, severity) DISCARD_MESSAGE
#define DBG_CHECK_GT(lhs, rhs, severity) DISCARD_MESSAGE
#endif

#if (defined(__GXX_EXPERIMENTAL_CXX0X__) || __cplusplus >= 201103L || defined(_MSC_VER))
#define MO_THROW_SPECIFIER noexcept(false)
#else
#define MO_THROW_SPECIFIER
#endif

#define MO_LOG_FIRST_N(severity, n)              \
    static int LOG_OCCURRENCES = 0;              \
    if (LOG_OCCURRENCES <= n) ++LOG_OCCURRENCES; \
    if (LOG_OCCURRENCES <= n) MO_LOG(severity)

#define THROW(severity) mo::ThrowOnDestroy_##severity(__FUNCTION__, __FILE__, __LINE__).stream()

#if (defined(__GXX_EXPERIMENTAL_CXX0X__) || __cplusplus >= 201103L || defined(_MSC_VER))
#define MO_THROW_SPECIFIER noexcept(false)
#else
#define MO_THROW_SPECIFIER
#endif

#define CUDA_ERROR_CHECK(exp) \
    cudaError_t err = exp;    \
    (err == cudaSuccess) ? (void)0 : mo::LogMessageVoidify() & THROW(warning) << "[" << cudaGetErrorString(err) << "]"

namespace mo {

MO_EXPORTS void InitLogging();

void MO_EXPORTS collectCallstack(size_t skipLevels, bool makeFunctionNamesStandOut, const std::function<void(const std::string&)>& write);
std::string MO_EXPORTS printCallstack(size_t skipLevels, bool makeFunctionNamesStandOut);
std::string MO_EXPORTS printCallstack(size_t skipLevels, bool makeFunctionNamesStandOut, std::stringstream& ss);

struct MO_EXPORTS ICallstackException {
    virtual ~ICallstackException();
    virtual std::string callstack() = 0;
};

// This class creates a message to be logged with
template <class Exc, boost::log::BOOST_LOG_VERSION_NAMESPACE::trivial::severity_level Severity>
struct CallstackSeverityException : virtual public Exc, virtual public ICallstackException {

    template <class... Args>
    CallstackSeverityException(Args... args)
        : Exc(std::forward(args)...)
        , _msg(s_msg_buffer) {}

    const char* what() const noexcept {
        return _msg.str().c_str();
    }

    std::string callstack() {
        return _callstack;
    }

    CallstackSeverityException& operator()(int error, const char* file, int line, const char* function) {
        _msg.str(std::string());
        _msg << file << ":" << line << " " << error << " in function [" << function << "]";

        _callstack = printCallstack(1, true);
        return *this;
    }

    template <class T>
    CallstackSeverityException& operator<<(const T& value) {
        _msg << value;
        return *this;
    }

protected:
    std::stringstream&  _msg;
    std::string         _callstack;
    static thread_local std::stringstream s_msg_buffer;
    static thread_local CallstackSeverityException<Exc, Severity> s_instance;
};
template <class Exc, boost::log::BOOST_LOG_VERSION_NAMESPACE::trivial::severity_level Severity>
thread_local CallstackSeverityException<Exc, Severity> CallstackSeverityException<Exc, Severity>::s_instance;

template <class Exc, boost::log::BOOST_LOG_VERSION_NAMESPACE::trivial::severity_level Severity>
thread_local std::stringstream CallstackSeverityException<Exc, Severity>::s_msg_buffer;

struct MO_EXPORTS IExceptionWithCallStackBase {
    virtual const char* callStack() const = 0;
    virtual ~IExceptionWithCallStackBase();
};

struct MO_EXPORTS ThrowStream : public IExceptionWithCallStackBase {
    ThrowStream(const ThrowStream& other);
    ThrowStream& operator()(int error, const char* file, int line, const char* func, bool collect_callstack = true);
    virtual const char* callStack() const;
    const char*         what() const noexcept;
    template <class T>
    ThrowStream& operator<<(const T& value) {
        msg << value;
        return *this;
    }
    static thread_local ThrowStream s_instance;

protected:
    ThrowStream();
    std::stringstream&  msg;
    std::string         callstack;
    static thread_local std::stringstream s_msg_buffer;
};

struct MO_EXPORTS ThrowStream_trace : public ThrowStream {
    ~ThrowStream_trace();
    ThrowStream_trace& operator()(int error, const char* file, int line, const char* func, bool collect_callstack = true);
};

struct MO_EXPORTS ThrowStream_debug : public ThrowStream {
    ~ThrowStream_debug();
    ThrowStream_debug& operator()(int error, const char* file, int line, const char* func, bool collect_callstack = true);
};

struct MO_EXPORTS ThrowStream_info : public ThrowStream {
    ~ThrowStream_info();
    ThrowStream_info& operator()(int error, const char* file, int line, const char* func, bool collect_callstack = true);
};

struct MO_EXPORTS ThrowStream_warning : public ThrowStream {
    ~ThrowStream_warning();
    ThrowStream_warning& operator()(int error, const char* file, int line, const char* func, bool collect_callstack = true);
};

struct MO_EXPORTS ThrowStream_error : public ThrowStream {
    ~ThrowStream_error();
    ThrowStream_error& operator()(int error, const char* file, int line, const char* func, bool collect_callstack = true);
};

class MO_EXPORTS ThrowOnDestroy {
public:
    ThrowOnDestroy(const char* function, const char* file, int line);
    std::ostringstream& stream();
    ~ThrowOnDestroy() MO_THROW_SPECIFIER;

protected:
    std::ostringstream log_stream_;
};

class MO_EXPORTS ThrowOnDestroy_trace : public ThrowOnDestroy {
public:
    ThrowOnDestroy_trace(const char* function, const char* file, int line);
    ~ThrowOnDestroy_trace() MO_THROW_SPECIFIER;
};

class MO_EXPORTS ThrowOnDestroy_debug : public ThrowOnDestroy {
public:
    ThrowOnDestroy_debug(const char* function, const char* file, int line);
};

class MO_EXPORTS ThrowOnDestroy_info : public ThrowOnDestroy {
public:
    ThrowOnDestroy_info(const char* function, const char* file, int line);
    ~ThrowOnDestroy_info() MO_THROW_SPECIFIER;
};

class MO_EXPORTS ThrowOnDestroy_warning : public ThrowOnDestroy {
public:
    ThrowOnDestroy_warning(const char* function, const char* file, int line);
    ~ThrowOnDestroy_warning() MO_THROW_SPECIFIER;
};

class MO_EXPORTS EatMessage {
public:
    EatMessage() {}
    std::stringstream& stream() {
        return eat;
    }

private:
    std::stringstream eat;
    EatMessage(const EatMessage&);
    void operator=(const EatMessage&);
};

class MO_EXPORTS LogMessageVoidify {
public:
    LogMessageVoidify() {}
    // This has to be an operator with a precedence lower than << but
    // higher than ?:
    void operator&(std::ostream&) {}
};

// Exception wrapper to include native call stack string
template <class E>
class ExceptionWithCallStack : public E, public IExceptionWithCallStackBase {
public:
    ExceptionWithCallStack(const std::string& msg, const std::string& callstack)
        : E(msg)
        , m_callStack(callstack) {}

    ExceptionWithCallStack(const E& exc, const std::string& callstack)
        : E(exc)
        , m_callStack(callstack) {}

    virtual const char* callStack() const override {
        return m_callStack.c_str();
    }

protected:
    std::string m_callStack;
};

template <>
class MO_EXPORTS ExceptionWithCallStack<std::string> : public std::string, public IExceptionWithCallStackBase {
public:
    ExceptionWithCallStack(const std::string& msg, const std::string& callstack);

    virtual const char* callStack() const override;

protected:
    std::string m_callStack;
};
} // namespace mo