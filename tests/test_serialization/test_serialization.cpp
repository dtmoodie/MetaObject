#define BOOST_TEST_MAIN
// clang-format off
#include <MetaObject/serialization/TextPolicy.hpp>
#include "MetaObject/MetaParameters.hpp"
#include "MetaObject/core/detail/Counter.hpp"
#include "MetaObject/logging/CompileLogger.hpp"
#include "MetaObject/object/IMetaObject.hpp"
#include "MetaObject/object/detail/MetaObjectMacros.hpp"
#include "MetaObject/params/MetaParam.hpp"
#include "MetaObject/params/ParamMacros.hpp"
#include "MetaObject/params/TInputParam.hpp"
#include "MetaObject/params/TParamPtr.hpp"
#include "MetaObject/params/buffers/BufferFactory.hpp"
#include "MetaObject/params/detail/MetaParamImpl.hpp"
#include "MetaObject/serialization/Policy.hpp"
#include "MetaObject/serialization/TextPolicy.hpp"
#include "MetaObject/serialization/memory.hpp"
#include "MetaObject/signals/TSignal.hpp"
#include "MetaObject/signals/detail/SignalMacros.hpp"
#include "MetaObject/signals/detail/SlotMacros.hpp"
#include "RuntimeObjectSystem/IObjectFactorySystem.h"
#include "RuntimeObjectSystem/RuntimeObjectSystem.h"
#include "RuntimeObjectSystem/shared_ptr.hpp"
#include "cereal/archives/portable_binary.hpp"
#include "cereal/archives/xml.hpp"
#include <MetaObject/params/Types.hpp>
#include <MetaObject/serialization/SerializationFactory.hpp>

#include <fstream>
#include <istream>
// clang-format on
#ifdef HAVE_OPENCV
#include <opencv2/core.hpp>
#endif
#ifdef _MSC_VER
#include <boost/test/unit_test.hpp>
#else
#define BOOST_TEST_MODULE __FILE__
#include <boost/test/included/unit_test.hpp>
#endif

#include <boost/thread.hpp>
#include <iostream>

using namespace mo;

struct serializable_object : public IMetaObject {
    MO_BEGIN(serializable_object);
    PARAM(int, test, 5)
    PARAM(int, test2, 6)
    MO_END;
};

BuildCallback* cb = nullptr;
MO_REGISTER_OBJECT(serializable_object);

#define ASSERT_SERIALIZABLE(TYPE) static_assert(mo::IO::Text::imp::stream_serializable<TYPE>::value, "Checking stream serializable for " #TYPE)

BOOST_AUTO_TEST_CASE(test_serialization) {
    ASSERT_SERIALIZABLE(std::string);
    ASSERT_SERIALIZABLE(size_t);
    BOOST_REQUIRE(!mo::IO::Text::imp::stream_serializable<rcc::shared_ptr<serializable_object> >::value);
}

BOOST_AUTO_TEST_CASE(serialize_manual_xml) {
    cb = new BuildCallback();
    mo::MetaParams::initialize();
    MetaObjectFactory::instance()->getObjectSystem()->SetupObjectConstructors(PerModuleInterface::GetInstance());
    rcc::shared_ptr<serializable_object> obj = serializable_object::create();
    {
        std::ofstream            ofs("test.xml");
        cereal::XMLOutputArchive archive(ofs);
        obj->test  = 10;
        obj->test2 = 100;
        archive(*(obj.get()));
    }
    {
        std::ifstream                        ifs("test.xml");
        cereal::XMLInputArchive              inar(ifs);
        rcc::shared_ptr<serializable_object> obj2 = serializable_object::create();
        inar(*(obj2.get()));
        BOOST_REQUIRE_EQUAL(obj->test, obj2->test);
        BOOST_REQUIRE_EQUAL(obj->test2, obj2->test2);
    }
}

BOOST_AUTO_TEST_CASE(serialize_manual_binary) {
    rcc::shared_ptr<serializable_object> obj = serializable_object::create();
    {
        std::ofstream               ofs("test.bin");
        cereal::BinaryOutputArchive archive(ofs);
        obj->test  = 10;
        obj->test2 = 100;
        archive(*(obj.get()));
    }
    {
        std::ifstream                        ifs("test.bin");
        cereal::BinaryInputArchive           inar(ifs);
        rcc::shared_ptr<serializable_object> obj2 = serializable_object::create();
        inar(*(obj2.get()));
        BOOST_REQUIRE_EQUAL(obj->test, obj2->test);
        BOOST_REQUIRE_EQUAL(obj->test2, obj2->test2);
    }
}

BOOST_AUTO_TEST_CASE(serialize_by_policy_xml) {
    rcc::shared_ptr<serializable_object> obj = serializable_object::create();
    obj->test                                = 14;
    obj->test2                               = 13;
    {
        std::ofstream ofs("test2.xml");
        SerializerFactory::Serialize(obj, ofs, SerializerFactory::xml_e);
    }
    {
        std::ifstream ifs("test2.xml");
        SerializerFactory::DeSerialize(obj.get(), ifs, SerializerFactory::xml_e);
    }
}

BOOST_AUTO_TEST_CASE(serialize_by_policy_binary) {
    rcc::shared_ptr<serializable_object> obj = serializable_object::create();
    obj->test                                = 14;
    obj->test2                               = 13;
    {
        std::ofstream ofs("test2.bin", std::ios::binary);
        SerializerFactory::Serialize(obj, ofs, SerializerFactory::Binary_e);
    }
    {
        std::ifstream ifs("test2.bin", std::ios::binary);
        SerializerFactory::DeSerialize(obj.get(), ifs, SerializerFactory::Binary_e);
        BOOST_REQUIRE_EQUAL(obj->test, 14);
        BOOST_REQUIRE_EQUAL(obj->test2, 13);
    }
}

BOOST_AUTO_TEST_CASE(deserialize_to_new_object_xml) {
    std::ifstream ifs("test2.xml");
    auto          obj = SerializerFactory::DeSerialize(ifs, SerializerFactory::xml_e);
    BOOST_REQUIRE(obj);
    auto T = obj.DynamicCast<serializable_object>();
    BOOST_REQUIRE(T);
    BOOST_REQUIRE_EQUAL(T->test, 14);
    BOOST_REQUIRE_EQUAL(T->test2, 13);
}

BOOST_AUTO_TEST_CASE(deserialize_to_new_object_binary) {
    std::ifstream ifs("test2.bin", std::ios::binary);
    auto          obj = SerializerFactory::DeSerialize(ifs, SerializerFactory::Binary_e);
    BOOST_REQUIRE(obj);
    auto T = obj.DynamicCast<serializable_object>();
    BOOST_REQUIRE(T);
    BOOST_REQUIRE_EQUAL(T->test, 14);
    BOOST_REQUIRE_EQUAL(T->test2, 13);
}

BOOST_AUTO_TEST_CASE(serialize_multi_by_policy_binary) {
    rcc::shared_ptr<serializable_object> obj1 = serializable_object::create();
    rcc::shared_ptr<serializable_object> obj2 = serializable_object::create();
    obj1->test                                = 14;
    obj1->test2                               = 13;
    obj2->test                                = 15;
    obj2->test2                               = 16;

    {
        std::ofstream ofs("test2.bin", std::ios::binary);
        SerializerFactory::Serialize(obj1.get(), ofs, SerializerFactory::Binary_e);
        SerializerFactory::Serialize(obj2.get(), ofs, SerializerFactory::Binary_e);
    }
    {
        std::ifstream ifs("test2.bin", std::ios::binary);
        auto          new_obj1 = rcc::shared_ptr<serializable_object>(SerializerFactory::DeSerialize(ifs, SerializerFactory::Binary_e));
        auto          new_obj2 = rcc::shared_ptr<serializable_object>(SerializerFactory::DeSerialize(ifs, SerializerFactory::Binary_e));
        BOOST_REQUIRE_EQUAL(new_obj1->test, 14);
        BOOST_REQUIRE_EQUAL(new_obj1->test2, 13);
        BOOST_REQUIRE_EQUAL(new_obj2->test, 15);
        BOOST_REQUIRE_EQUAL(new_obj2->test2, 16);
    }
}

INSTANTIATE_META_PARAM(mo::ReadFile);
INSTANTIATE_META_PARAM(std::vector<int>);
BOOST_AUTO_TEST_CASE(deserialize_text_path) {
    mo::ReadFile                data;
    mo::TParamPtr<mo::ReadFile> param;
    param.updatePtr(&data);
    auto deserialization_function = mo::SerializationFactory::instance()->getTextDeSerializationFunction(param.getTypeInfo());
    BOOST_REQUIRE(deserialization_function);
    std::stringstream ss;
    ss << "/asdf/asdf/asdf/test.txt";
    deserialization_function(&param, ss);
    BOOST_REQUIRE(data == "/asdf/asdf/asdf/test.txt");
}

BOOST_AUTO_TEST_CASE(serialize_text_vector) {
    std::vector<int>                 data = { 0, 1, 2, 3, 4, 5, 6, 7 };
    mo::TParamPtr<std::vector<int> > param;
    param.updatePtr(&data);
    auto serialization_function = mo::SerializationFactory::instance()->getTextSerializationFunction(param.getTypeInfo());
    BOOST_REQUIRE(serialization_function);
    std::stringstream ss;
    serialization_function(&param, ss);
    std::string str = ss.str();
    BOOST_REQUIRE(str == "8[0, 1, 2, 3, 4, 5, 6, 7]");
}

BOOST_AUTO_TEST_CASE(deserialize_text_vector) {
    std::vector<int>                 data;
    mo::TParamPtr<std::vector<int> > param;
    param.updatePtr(&data);
    auto deserialization_function = mo::SerializationFactory::instance()->getTextDeSerializationFunction(param.getTypeInfo());
    BOOST_REQUIRE(deserialization_function);
    std::stringstream ss;
    ss << "[0, 1, 2, 3, 4, 5, 6, 7]";
    deserialization_function(&param, ss);
    BOOST_REQUIRE(data.size() == 8);
    for (int expected_value = 0; expected_value < 8; ++expected_value) {
        BOOST_REQUIRE_EQUAL(data[expected_value], expected_value);
    }
}

BOOST_AUTO_TEST_CASE(serialize_unique_instance) {
    {
        auto inst1 = serializable_object::create();
        auto inst2 = inst1;
        mo::StartSerialization();
        std::ofstream             ofs("test_unique_instance.json");
        cereal::JSONOutputArchive ar(ofs);
        ar(CEREAL_NVP(inst1));
        ar(CEREAL_NVP(inst2));
        mo::EndSerialization();
    }
    {
        mo::StartSerialization();
        std::ifstream                        ifs("test_unique_instance.json");
        cereal::JSONInputArchive             ar(ifs);
        rcc::shared_ptr<serializable_object> inst1;
        rcc::shared_ptr<serializable_object> inst2;
        ar(CEREAL_NVP(inst1));
        ar(CEREAL_NVP(inst2));
        mo::EndSerialization();
        BOOST_REQUIRE_EQUAL(inst1, inst2);
    }
}

BOOST_AUTO_TEST_CASE(cleanup) {
    delete cb;
}
