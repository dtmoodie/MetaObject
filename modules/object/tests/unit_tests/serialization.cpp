#include "Objects.hpp"
#include <RuntimeObjectSystem/SimpleSerializer/SimpleSerializer.h>

#include <MetaObject/serialization/BinaryLoader.hpp>
#include <MetaObject/serialization/BinarySaver.hpp>
#include <MetaObject/serialization/JSONPrinter.hpp>

#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>

#include <fstream>
#include <istream>

#include <boost/test/auto_unit_test.hpp>
#include <boost/test/test_tools.hpp>

#include <boost/thread.hpp>
#include <iostream>

using namespace mo;
using namespace test;

BOOST_AUTO_TEST_CASE(serialization_of_params)
{
    auto inst1 = SerializableObject::create();

    inst1->test = 10;
    inst1->test2 = 20;

    // Do some serialization stuffs here
    SimpleSerializer serializer;
    serializer.SetIsLoading(false);
    serializer.Serialize(inst1.get());

    inst1->test = 20;
    inst1->test2 = 40;
    serializer.SetIsLoading(true);
    serializer.Serialize(inst1.get());
    BOOST_REQUIRE_EQUAL(inst1->test, 10);
    BOOST_REQUIRE_EQUAL(inst1->test2, 20);
}

BOOST_AUTO_TEST_CASE(statically_typed_serialization_json)
{
    auto inst = SerializableObject::create();
    inst->test = 100;
    inst->test2 = 200;
    std::stringstream ss;
    {
        mo::JSONSaver saver(ss);
        inst->save(saver);
    }

    ss.seekg(0);
    std::cout << ss.str() << std::endl;
    inst->test = 0;
    inst->test2 = 0;
    {
        mo::JSONLoader loader(ss);
        inst->load(loader);
    }

    BOOST_REQUIRE_EQUAL(inst->test, 100);
    BOOST_REQUIRE_EQUAL(inst->test2, 200);
}

BOOST_AUTO_TEST_CASE(statically_typed_serialization_binary)
{
    auto inst = SerializableObject::create();
    std::stringstream ss;
    {
        mo::BinarySaver saver(ss);
        inst->save(saver);
    }
    ss.seekg(0);

    inst->test = 0;
    inst->test2 = 0;
    {
        mo::BinaryLoader loader(ss);
        inst->load(loader);
    }
    BOOST_REQUIRE_EQUAL(inst->test, 5);
    BOOST_REQUIRE_EQUAL(inst->test2, 6);
}
