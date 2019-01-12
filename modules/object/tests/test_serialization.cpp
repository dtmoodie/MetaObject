// clang-format off
#include "MetaObject/core/detail/Counter.hpp"
#include "MetaObject/core.hpp"
#include "MetaObject/logging/CompileLogger.hpp"

#include <MetaObject/params/MetaParam.hpp>
#include <MetaObject/params/ParamMacros.hpp>
#include <MetaObject/params/TInputParam.hpp>
#include <MetaObject/params/TParamPtr.hpp>
#include <MetaObject/params/buffers/BufferFactory.hpp>
#include <MetaObject/params/detail/MetaParamImpl.hpp>

#include <MetaObject/object.hpp>
#include <MetaObject/serialization/memory.hpp>

#include "MetaObject/signals/TSignal.hpp"
#include "MetaObject/signals/detail/SignalMacros.hpp"
#include "MetaObject/signals/detail/SlotMacros.hpp"

#include "RuntimeObjectSystem/IObjectFactorySystem.h"
#include "RuntimeObjectSystem/RuntimeObjectSystem.h"
#include "RuntimeObjectSystem/shared_ptr.hpp"

#include "cereal/archives/portable_binary.hpp"
#include "cereal/archives/xml.hpp"

#include <MetaObject/types/file_types.hpp>
#include <MetaObject/serialization/SerializationFactory.hpp>
#include <MetaObject/serialization/CerealParameters.hpp>
#include <MetaObject/runtime_reflection/visitor_traits/vector.hpp>


#include <fstream>
#include <istream>
// clang-format on
#ifdef HAVE_OPENCV
#include <opencv2/core.hpp>
#endif

#include <boost/test/test_tools.hpp>
#include <boost/test/auto_unit_test.hpp>

#include <boost/thread.hpp>
#include <iostream>

using namespace mo;
struct SerializableObject : public MetaObject
{
    ~SerializableObject() override;
    MO_BEGIN(SerializableObject)
    PARAM(int, test, 5)
    PARAM(int, test2, 6)
    MO_END
};

SerializableObject::~SerializableObject()
{
}

MO_REGISTER_OBJECT(SerializableObject);

template <class TYPE, class Resetter, class Setter, class Checker>
void testParamSerialization(mo::IParam* param, const Resetter& resetter, const Setter& setter, const Checker& checkker)
{
    using OutputArchive = typename mo::ArchivePairs<TYPE>::Output;
    using InputArchive = typename mo::ArchivePairs<TYPE>::Input;
    resetter();
    {
        std::ofstream ofs("test");
        OutputArchive ar(ofs);
        auto serializer = mo::SerializationFactory::instance()->getSerializer<OutputArchive>(param->getTypeInfo());
        if (!serializer)
        {
            std::cout << "Missing serializer for " << mo::TypeTable::instance().typeToName(param->getTypeInfo()) << '\n';
            std::cout << "Have serializers for \n";
            const auto serializers = mo::SerializationFactory::instance()->listSerializableTypes();
            for (const auto& type : serializers)
            {
                std::cout << "  " << mo::TypeTable::instance().typeToName(type) << '\n';
            }
            std::cout << std::endl;
        }
        BOOST_REQUIRE(serializer);
        serializer(param, ar);
    }
    setter();
    {
        std::ifstream ifs("test");
        InputArchive ar(ifs);
        auto deserializer = mo::SerializationFactory::instance()->getDeserializer<InputArchive>(param->getTypeInfo());
        BOOST_REQUIRE(deserializer);
        deserializer(param, ar);
    }
    checkker();
}

template <class Resetter, class Setter, class Checker>
void testAllSerialization(mo::IParam* param, const Resetter& resetter, const Setter& setter, const Checker& checker)
{
    testParamSerialization<mo::Binary>(param, resetter, setter, checker);
    testParamSerialization<mo::JSON>(param, resetter, setter, checker);
    testParamSerialization<mo::XML>(param, resetter, setter, checker);
}

template <class T, class Resetter, class Setter, class Checker>
void testParamSerialization(T& data, const Resetter& resetter, const Setter& setter, const Checker& checker)
{
    mo::TParamPtr<T> param;
    param.setName("param");
    param.updatePtr(&data);
    testAllSerialization(&param, resetter, setter, checker);
}

template <class T>
void testScalar()
{
    T data = 0;
    testParamSerialization(
        data, [&data]() { data = 0; }, [&data]() { data = 5; }, [&data]() { BOOST_REQUIRE_EQUAL(data, 0); });
}

template <class T>
void testVector()
{
    std::vector<T> data(10);
    testParamSerialization(data,
                           [&data]() {
                               for (size_t i = 0; i < data.size(); ++i)
                               {
                                   data[i] = i;
                               }

                           },
                           [&data]() {
                               for (size_t i = 0; i < data.size(); ++i)
                               {
                                   data[i] = i + 5;
                               }
                           },
                           [&data]() {
                               for (size_t i = 0; i < data.size(); ++i)
                               {
                                   BOOST_REQUIRE_EQUAL(data[i], i);
                               }

                           });
}

BOOST_AUTO_TEST_CASE(serialize_parameter)
{
    testScalar<float>();
    testScalar<double>();
    testScalar<unsigned char>();
    testScalar<char>();
    testScalar<int>();
    testScalar<unsigned int>();

    testVector<float>();
    testVector<double>();
    testVector<unsigned char>();
    testVector<char>();
    testVector<int>();
    testVector<unsigned int>();
}

BOOST_AUTO_TEST_CASE(serialize_manual_xml)
{
    auto obj = SerializableObject::create();
    {
        std::ofstream ofs("test.xml");
        cereal::XMLOutputArchive archive(ofs);
        obj->test = 10;
        obj->test2 = 100;
        archive(*(obj.get()));
    }
    {
        std::ifstream ifs("test.xml");
        cereal::XMLInputArchive inar(ifs);
        auto obj2 = SerializableObject::create();
        inar(*(obj2.get()));
        BOOST_REQUIRE_EQUAL(obj->test, obj2->test);
        BOOST_REQUIRE_EQUAL(obj->test2, obj2->test2);
    }
}

BOOST_AUTO_TEST_CASE(serialize_manual_binary)
{
    auto obj = SerializableObject::create();
    {
        std::ofstream ofs("test.bin");
        cereal::BinaryOutputArchive archive(ofs);
        obj->test = 10;
        obj->test2 = 100;
        archive(*(obj.get()));
    }
    {
        std::ifstream ifs("test.bin");
        cereal::BinaryInputArchive inar(ifs);
        auto obj2 = SerializableObject::create();
        inar(*(obj2.get()));
        BOOST_REQUIRE_EQUAL(obj->test, obj2->test);
        BOOST_REQUIRE_EQUAL(obj->test2, obj2->test2);
    }
}

BOOST_AUTO_TEST_CASE(serialize_manual_json)
{
    auto obj = SerializableObject::create();
    {
        std::ofstream ofs("test.json");
        cereal::JSONOutputArchive archive(ofs);
        obj->test = 10;
        obj->test2 = 100;
        archive(*(obj.get()));
    }
    {
        std::ifstream ifs("test.json");
        cereal::JSONInputArchive inar(ifs);
        auto obj2 = SerializableObject::create();
        inar(*(obj2.get()));
        BOOST_REQUIRE_EQUAL(obj->test, obj2->test);
        BOOST_REQUIRE_EQUAL(obj->test2, obj2->test2);
    }
}

BOOST_AUTO_TEST_CASE(serialize_create_json)
{
    auto obj = SerializableObject::create();
    {
        std::ofstream ofs("test.json");
        cereal::JSONOutputArchive archive(ofs);
        obj->test = 10;
        obj->test2 = 100;
        archive(obj);
    }
    {
        std::ifstream ifs("test.json");
        cereal::JSONInputArchive inar(ifs);
        rcc::shared_ptr<SerializableObject> obj2;
        inar(obj2);
        BOOST_REQUIRE(obj2);
        BOOST_REQUIRE_EQUAL(obj->test, obj2->test);
        BOOST_REQUIRE_EQUAL(obj->test2, obj2->test2);
    }
}
