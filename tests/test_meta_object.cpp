#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN

#include "MetaObject/IMetaObject.hpp"
#include "MetaObject/Signals/TypedSignal.hpp"

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "MetaObject"
#include <boost/test/unit_test.hpp>

using namespace mo;

BOOST_AUTO_TEST_CASE(test_meta_object1)
{
    class test_meta_object1: public IMetaObject
    {
    public:
        test_meta_object1()
        {

        }
        TypedSignal<void(int)> int_signal;
    };
    test_meta_object1 obj;
    
}