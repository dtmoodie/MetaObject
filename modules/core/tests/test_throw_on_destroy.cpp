#include "MetaObject/logging/logging.hpp"

#include <boost/stacktrace.hpp>

#include <gtest/gtest.h>


#include <iostream>
#include <sstream>
using namespace mo;

bool throwing_function()
{
    THROW(debug, "throwing from function");
    return false;
}

struct Throw_debug : public std::exception
{
    Throw_debug()
        : msg(s_msg_buffer)
    {
    }

    Throw_debug(const Throw_debug& other)
        : msg(other.msg)
        , callstack(other.callstack)
    {
    }

    Throw_debug& operator()(int error, const char* file, int line, const char* function, bool collect_callstack = true)
    {
        msg.str(std::string());
        msg << file << ":" << line << " " << error << " in function [" << function << "]";
        if (collect_callstack)
        {
            boost::stacktrace::stacktrace st;
            msg << st;
        }
        return *this;
    }

    const char* what() const noexcept
    {
        return msg.str().c_str();
    }

    template <class T>
    Throw_debug& operator<<(const T& value)
    {
        msg << value;
        return *this;
    }
    std::stringstream& msg;
    std::string callstack;
    static thread_local std::stringstream s_msg_buffer;
    static thread_local Throw_debug s_instance;
};

thread_local Throw_debug Throw_debug::s_instance;
thread_local std::stringstream Throw_debug::s_msg_buffer;

void test_throw_obj()
{
    throw Throw_debug::s_instance(0, __FILE__, __LINE__, __FUNCTION__) << "asdfkjasdf";
}

TEST(throw_on_destroy, throw_obj)
{
    ASSERT_THROW(test_throw_obj(), Throw_debug);
}

TEST(signals, signals_test)
{
    ASSERT_THROW(THROW(debug, "test"), mo::TExceptionWithCallstack<std::runtime_error>);
    ASSERT_THROW(MO_ASSERT_EQ(throwing_function(), true), mo::TExceptionWithCallstack<std::runtime_error>);
}
