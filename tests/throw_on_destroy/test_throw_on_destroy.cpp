
#define BOOST_TEST_MAIN

#include "MetaObject/logging/logging.hpp"

#ifdef _MSC_VER
#include <boost/test/unit_test.hpp>
#else
#define BOOST_TEST_MODULE __FILE__
#include <boost/test/included/unit_test.hpp>
#endif
#include <iostream>
#include <sstream>
using namespace mo;

bool throwing_function()
{
    THROW(debug) << "throwing from function";
    return false;
}

struct Throw_debug : public std::exception
{
    Throw_debug() : msg(s_msg_buffer) {}

    Throw_debug(const Throw_debug& other) : msg(other.msg), callstack(other.callstack) {}

    Throw_debug& operator()(int error, const char* file, int line, const char* function, bool collect_callstack = true)
    {
        msg.str(std::string());
        msg << file << ":" << line << " " << error << " in function [" << function << "]";
        if (collect_callstack)
            callstack = printCallstack(1, true);
        return *this;
    }

    const char* what() const noexcept { return msg.str().c_str(); }

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

BOOST_AUTO_TEST_CASE(throw_obj)
{
    BOOST_REQUIRE_THROW(test_throw_obj(), Throw_debug);
}

BOOST_AUTO_TEST_CASE(signals_test)
{
    BOOST_REQUIRE_THROW(THROW(debug) << "test", mo::ExceptionWithCallStack<std::string>);
    BOOST_REQUIRE_THROW(MO_ASSERT_EQ(throwing_function(), true), mo::ExceptionWithCallStack<std::string>);
}
