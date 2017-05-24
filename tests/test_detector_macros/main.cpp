#define BOOST_TEST_MAIN

#ifdef _MSC_VER
#include <boost/test/unit_test.hpp>
#else
#define BOOST_TEST_MODULE "HelperMacros"
#include <boost/test/included/unit_test.hpp>
#endif
#include <boost/thread.hpp>
#include <iostream>
#include <MetaObject/core/detail/HelperMacros.hpp>


struct object_with_members
{
    int member_variable;
    int foo;
};

struct object_with_function
{
    static int foo()
    {
        return 1;
    }
    static int bar(int value)
    {
        return value;
    }
};

struct object_with_different_signature
{
    static int foo(int val)
    {
        return 2;
    }
};


template<class T> struct call_foo
{
    DEFINE_HAS_STATIC_FUNCTION(HasFoo, foo, int(*)(void));
    template<class U>
    static int helper(typename std::enable_if<HasFoo<U>::value, void>::type* = 0)
    {
        return T::foo();
    }
    template<class U>
    static int helper(typename std::enable_if<!HasFoo<U>::value, void>::type* = 0)
    {
        return 0;
    }

    static int call()
    {
        return helper<T>();
    }
};

template<class T> struct call_bar
{
    DEFINE_HAS_STATIC_FUNCTION(HasBar, bar, int(*)(int));
    template<class U>
    static int helper(typename std::enable_if<HasBar<U>::value, int>::type value)
    {
        return T::bar(value);
    }
    template<class U>
    static int helper(typename std::enable_if<!HasBar<U>::value, int>::type value)
    {
        return 0;
    }

    static int call(int value)
    {
        return helper<T>(value);
    }
};




DEFINE_MEMBER_DETECTOR(member_variable);


BOOST_AUTO_TEST_CASE(test_detect_member)
{
    BOOST_REQUIRE(Detect_member_variable<object_with_members>::value);
    BOOST_REQUIRE(!Detect_member_variable<object_with_function>::value);
}

BOOST_AUTO_TEST_CASE(test_detect_static_function)
{
    BOOST_REQUIRE_EQUAL(call_foo<object_with_members>::call(), 0);
    BOOST_REQUIRE_EQUAL(call_foo<object_with_function>::call(), 1);
    BOOST_REQUIRE_EQUAL(call_foo<object_with_different_signature>::call(), 0);

    BOOST_REQUIRE_EQUAL(call_bar<object_with_members>::call(10), 0);
    BOOST_REQUIRE_EQUAL(call_bar<object_with_function>::call(10), 10);
    BOOST_REQUIRE_EQUAL(call_bar<object_with_different_signature>::call(10), 0);
}
