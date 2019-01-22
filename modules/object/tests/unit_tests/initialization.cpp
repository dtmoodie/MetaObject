#include "Objects.hpp"
#include <boost/test/auto_unit_test.hpp>
#include <boost/test/test_tools.hpp>

using namespace test;

BOOST_AUTO_TEST_CASE(constructor)
{
    {
        auto inst = mo::MetaObjectFactory::instance()->create("SerializableObject");
        BOOST_REQUIRE_NE(inst, nullptr);
    }
    {
        auto inst = SerializableObject::create();
        BOOST_REQUIRE_NE(inst, nullptr);
    }
}


BOOST_AUTO_TEST_CASE(initialization)
{
    auto inst = SerializableObject::create();
    BOOST_REQUIRE_NE(inst, nullptr);
    BOOST_REQUIRE_EQUAL(inst->test, 5);
    BOOST_REQUIRE_EQUAL(inst->test2, 6);
}

BOOST_AUTO_TEST_CASE(derived_initialization)
{
    auto derived_obj = DerivedParams::create();
    BOOST_REQUIRE_EQUAL(derived_obj->base_param, 5);
    BOOST_REQUIRE_EQUAL(derived_obj->derived_param, 10);
    derived_obj->base_param = 10;
    derived_obj->derived_param = 100;
    derived_obj->initParams(true);
    BOOST_REQUIRE_EQUAL(derived_obj->base_param, 5);
    BOOST_REQUIRE_EQUAL(derived_obj->derived_param, 10);
}
