#pragma once

#include "MetaObject/Detail/Export.hpp"
#include "MetaObject/Logging/LogHelperMacros.hpp"

#include <boost/log/trivial.hpp>
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
#include <execinfo.h>
#include <cxxabi.h>
#endif

#ifdef LOG
#undef LOG
#endif
#ifdef CHECK_EQ
#undef CHECK_EQ
#endif
#ifdef CHECK_NE
#undef CHECK_NE
#endif
#ifdef CHECK_LE
#undef CHECK_LE
#endif
#ifdef CHECK_LT
#undef CHECK_LT
#endif
#ifdef CHECK_GE
#undef CHECK_GE
#endif
#ifdef CHECK_GT
#undef CHECK_GT
#endif
#ifdef CHECK_OP
#undef CHECK_OP
#endif
#ifdef ASSERT_OP
#undef ASSERT_OP
#endif
#ifdef LOG_FIRST_N
#undef LOG_FIRST_N
#endif
#ifdef CHECK_OP
#undef CHECK_OP
#endif
#ifdef CHECK_GT
#undef CHECK_GT
#endif
#ifdef CHECK_GE
#undef CHECK_GE
#endif
#ifdef CHECK_LT
#undef CHECK_LT
#endif
#ifdef CHECK_LE
#undef CHECK_LE
#endif
#ifdef CHECK_NE
#undef CHECK_NE
#endif
#ifdef DISCARD_MESSAGE
#undef DISCARD_MESSAGE
#endif

#define DISCARD_MESSAGE true ? (void)0 : mo::LogMessageVoidify() & mo::eat_message().stream()

#define LOG_EVERY_N(severity, n) \
    static int LOG_OCCURRENCES = 0, LOG_OCCURRENCES_MOD_N = 0; \
    ++LOG_OCCURRENCES; \
    if (++LOG_OCCURRENCES_MOD_N > n) LOG_OCCURRENCES_MOD_N -= n; \
    if (LOG_OCCURRENCES_MOD_N == 1) \
        LOG(severity)

#define LOG_FIRST_N(severity, n) static int LOG_OCCURRENCES = 0; if(LOG_OCCURRENCES <= n) ++LOG_OCCURRENCES; if(LOG_OCCURRENCES <= n) LOG(severity)
#define MO_LOG(severity) BOOST_LOG_TRIVIAL(severity) << "[" << this->GetTypeName() << "::" __FUNCTION__ "] "
#define LOG(severity) BOOST_LOG_TRIVIAL(severity) << "[" << __FUNCTION__ << "] "
#define DOIF(condition, expression, severity) if(condition) { LOG(severity) << #condition << " is true, thus performing " << #expression; expression;} else { LOG(severity) << #condition << " failed";}

#define DOIF_LOG_PASS(condition, expression, severity) if(condition) { LOG(severity) << #condition << " is true, thus performing " << #expression; expression;}
#define DOIF_LOG_FAIL(condition, expression, severity) if(condition) { expression; } else { LOG(severity) << "Unable to perform " #expression " due to " #condition << " failed";}

#define LOGIF_EQ(lhs, rhs, severity) if(lhs == rhs)  LOG(severity) << "if(" << #lhs << " == " << #rhs << ")" << "[" << lhs << " == " << rhs << "]";
#define LOGIF_NEQ(lhs, rhs, severity) if(lhs != rhs) LOG(severity) << "if(" << #lhs << " != " << #rhs << ")" << "[" << lhs << " != " << rhs << "]";

#define ASSERT_OP(op, lhs, rhs) if(!(lhs op rhs)) mo::ThrowOnDestroy(__FUNCTION__, __FILE__, __LINE__).stream() << "[" << #lhs << " " << #op << " " << #rhs << "] - Failed (" << lhs << " " <<  #op << " " << rhs << ")"

#define MO_ASSERT(exp) if( !(exp) ) mo::ThrowOnDestroy(__FUNCTION__, __FILE__, __LINE__).stream() << "[" << #exp << "] failed!"

#define ASSERT_EQ(lhs, rhs)  ASSERT_OP(==, lhs, rhs)
#define ASSERT_NE(lhs, rhs)  ASSERT_OP(!=, lhs, rhs)
#define ASSERT_LE(lhs, rhs)  ASSERT_OP(<=, lhs, rhs)
#define ASSERT_LT(lhs, rhs)  ASSERT_OP(< , lhs, rhs)
#define ASSERT_GE(lhs, rhs)  ASSERT_OP(>=, lhs, rhs)
#define ASSERT_GT(lhs, rhs)  ASSERT_OP(> , lhs, rhs)

#define CHECK_OP(op, lhs, rhs, severity) if(lhs op rhs)  LOG(severity) << "[" << #lhs << " " << #op << " " << #rhs << "] - Failed (" << lhs << " " <<  #op << " " << rhs << ")"

#define CHECK_EQ(lhs, rhs, severity) CHECK_OP(==, lhs, rhs, severity)
#define CHECK_NE(lhs, rhs, severity) CHECK_OP(!=, lhs, rhs, severity)
#define CHECK_LE(lhs, rhs, severity) CHECK_OP(<=, lhs, rhs, severity)
#define CHECK_LT(lhs, rhs, severity) CHECK_OP(< , lhs, rhs, severity)
#define CHECK_GE(lhs, rhs, severity) CHECK_OP(>=, lhs, rhs, severity)
#define CHECK_GT(lhs, rhs, severity) CHECK_OP(> , lhs, rhs, severity)

#ifdef _DEBUG
#define DBG_CHECK_EQ(lhs, rhs, severity) CHECK_OP(==, lhs, rhs, severity)
#define DBG_CHECK_NE(lhs, rhs, severity) CHECK_OP(!=, lhs, rhs, severity)
#define DBG_CHECK_LE(lhs, rhs, severity) CHECK_OP(<=, lhs, rhs, severity)
#define DBG_CHECK_LT(lhs, rhs, severity) CHECK_OP(< , lhs, rhs, severity)
#define DBG_CHECK_GE(lhs, rhs, severity) CHECK_OP(>=, lhs, rhs, severity)
#define DBG_CHECK_GT(lhs, rhs, severity) CHECK_OP(> , lhs, rhs, severity)
#else
#define DBG_CHECK_EQ(lhs, rhs, severity) DISCARD_MESSAGE
#define DBG_CHECK_NE(lhs, rhs, severity) DISCARD_MESSAGE
#define DBG_CHECK_LE(lhs, rhs, severity) DISCARD_MESSAGE
#define DBG_CHECK_LT(lhs, rhs, severity) DISCARD_MESSAGE
#define DBG_CHECK_GE(lhs, rhs, severity) DISCARD_MESSAGE
#define DBG_CHECK_GT(lhs, rhs, severity) DISCARD_MESSAGE
#endif

#if (defined(__GXX_EXPERIMENTAL_CXX0X__) ||\
        __cplusplus >= 201103L || defined(_MSC_VER))
#define MO_THROW_SPECIFIER noexcept(false)
#else
#define MO_THROW_SPECIFIER
#endif

#define LOG_FIRST_N(severity, n) static int LOG_OCCURRENCES = 0; if(LOG_OCCURRENCES <= n) ++LOG_OCCURRENCES; if(LOG_OCCURRENCES <= n) LOG(severity)

#define THROW(severity) mo::ThrowOnDestroy_##severity(__FUNCTION__, __FILE__, __LINE__).stream()

#if (defined(__GXX_EXPERIMENTAL_CXX0X__) || \
        __cplusplus >= 201103L || defined(_MSC_VER))
#define MO_THROW_SPECIFIER noexcept(false)
#else
#define MO_THROW_SPECIFIER
#endif

namespace mo {

MO_EXPORTS void InitLogging();

void MO_EXPORTS collectCallstack(size_t skipLevels, bool makeFunctionNamesStandOut, const std::function<void(const std::string&)>& write);
std::string MO_EXPORTS printCallstack(size_t skipLevels, bool makeFunctionNamesStandOut);
std::string MO_EXPORTS printCallstack(size_t skipLevels, bool makeFunctionNamesStandOut, std::stringstream& ss);


struct ICallstackException{
    virtual ~ICallstackException();
    virtual std::string callstack();
};

// This class creates a message to be logged with 
template<class Exc, boost::log::BOOST_LOG_VERSION_NAMESPACE::trivial::severity_level Severity>
struct CallstackSeverityException: virtual public Exc, virtual public ICallstackException{
    template<class... Args>
    CallstackSeverityException(Args... args):
        Exc(std::forward(args)...), _msg(s_msg_buffer){}

    const char* what() const noexcept{
        return _msg.str().c_str();
    }

    std::string callstack(){
        return _callstack;
    }

    CallstackSeverityException& operator()(int error, const char* file, int line, const char* function){
        _msg.str(std::string());
        _msg << file << ":" << line << " " << error << " in function [" << function << "]";
        
        _callstack = printCallstack(1, true);
        return *this;
    }

    template<class T>
    CallstackSeverityException& operator<<(const T& value){
        _msg << value;
        return *this;
    }

protected:
    std::stringstream& _msg;
    std::string _callstack;
    static thread_local std::stringstream s_msg_buffer;
    static thread_local CallstackSeverityException<Exc, Severity> s_instance;
};
template<class Exc, boost::log::BOOST_LOG_VERSION_NAMESPACE::trivial::severity_level Severity>
thread_local CallstackSeverityException<Exc, Severity> CallstackSeverityException<Exc, Severity>::s_instance;

template<class Exc, boost::log::BOOST_LOG_VERSION_NAMESPACE::trivial::severity_level Severity>
thread_local std::stringstream CallstackSeverityException<Exc, Severity>::s_msg_buffer;


class MO_EXPORTS ThrowOnDestroy {
public:
    ThrowOnDestroy(const char* function, const char* file, int line);
    std::ostringstream &stream();
    ~ThrowOnDestroy() MO_THROW_SPECIFIER;

protected:
    std::ostringstream log_stream_;
};

class MO_EXPORTS ThrowOnDestroy_trace: public ThrowOnDestroy {
public:
    ThrowOnDestroy_trace(const char* function, const char* file, int line);
    ~ThrowOnDestroy_trace() MO_THROW_SPECIFIER;
};

class MO_EXPORTS ThrowOnDestroy_debug: public ThrowOnDestroy {
public:
    ThrowOnDestroy_debug(const char* function, const char* file, int line);
};

class MO_EXPORTS ThrowOnDestroy_info: public ThrowOnDestroy {
public:
    ThrowOnDestroy_info(const char* function, const char* file, int line);
    ~ThrowOnDestroy_info() MO_THROW_SPECIFIER;
};

class MO_EXPORTS ThrowOnDestroy_warning: public ThrowOnDestroy {
public:
    ThrowOnDestroy_warning(const char* function, const char* file, int line);
    ~ThrowOnDestroy_warning() MO_THROW_SPECIFIER;
};

class MO_EXPORTS EatMessage {
public:
    EatMessage() {}
    std::stringstream &stream() {
        return eat;
    }
private:
    std::stringstream eat;
    EatMessage(const EatMessage&);
    void operator=(const EatMessage&);
};

class MO_EXPORTS LogMessageVoidify {
public:
    LogMessageVoidify() { }
    // This has to be an operator with a precedence lower than << but
    // higher than ?:
    void operator&(std::ostream&) { }
};


struct MO_EXPORTS IExceptionWithCallStackBase {
    virtual const char * CallStack() const = 0;
    virtual ~IExceptionWithCallStackBase();
};

// Exception wrapper to include native call stack string
template <class E>
class ExceptionWithCallStack : public E, public IExceptionWithCallStackBase {
public:
    ExceptionWithCallStack(const std::string& msg, const std::string& callstack) :
        E(msg), m_callStack(callstack){ }

    ExceptionWithCallStack(const E& exc, const std::string& callstack) :
        E(exc), m_callStack(callstack){ }

    virtual const char * CallStack() const override {
        return m_callStack.c_str();
    }

protected:
    std::string m_callStack;
};

template<> class ExceptionWithCallStack<std::string>: public std::string, public IExceptionWithCallStackBase {
public:
    ExceptionWithCallStack(const std::string& msg, const std::string& callstack) :
        std::string(msg), m_callStack(callstack)
    { }

    virtual const char * CallStack() const override {
        return m_callStack.c_str();
    }

protected:
    std::string m_callStack;
};
}
