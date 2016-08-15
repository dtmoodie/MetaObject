#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include "MetaObject/Parameters/MetaParameter.hpp"
#include "MetaObject/IMetaObject.hpp"
#include "MetaObject/Signals/TypedSignal.hpp"
#include "MetaObject/Detail/Counter.hpp"
#include "MetaObject/Detail/MetaObjectMacros.hpp"
#include "MetaObject/Signals/detail/SignalMacros.hpp"
#include "MetaObject/Signals/detail/SlotMacros.hpp"
#include "MetaObject/Parameters//ParameterMacros.hpp"
#include "MetaObject/Parameters/TypedParameterPtr.hpp"
#include "MetaObject/Parameters/TypedInputParameter.hpp"
#include "MetaObject/Logging/CompileLogger.hpp"
#include "MetaObject/Parameters/Buffers/BufferFactory.hpp"
#include "MetaObject/IO/Policy.hpp"
#include "shared_ptr.hpp"
#include "RuntimeObjectSystem.h"
#include "shared_ptr.hpp"
#include "IObjectFactorySystem.h"
#include "cereal/archives/xml.hpp"
#include "cereal/archives/portable_binary.hpp"
#include <fstream>

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "parameter"
#include <boost/test/unit_test.hpp>
#include <boost/thread.hpp>
#include <iostream>

using namespace mo;

struct serializable_object: public IMetaObject
{
    MO_BEGIN(serializable_object);
    PARAM(int, test, 5);
    PARAM(int, test2, 6);
    MO_END;
};



BuildCallback* cb = nullptr;
MO_REGISTER_OBJECT(serializable_object);

BOOST_AUTO_TEST_CASE(serialize_manual_xml)
{
    cb = new BuildCallback();
    MetaObjectFactory::Instance()->GetObjectSystem()->SetupObjectConstructors(PerModuleInterface::GetInstance());
    rcc::shared_ptr<serializable_object> obj = serializable_object::Create();
    {
        std::ofstream ofs("test.xml");
        cereal::XMLOutputArchive archive(ofs);
        obj->test = 10;
        obj->test2 = 100;
        archive(*(obj.Get()));
    }
    {
        std::ifstream ifs("test.xml");
        cereal::XMLInputArchive inar(ifs);
        rcc::shared_ptr<serializable_object> obj2 = serializable_object::Create();
        inar(*(obj2.Get()));
        BOOST_REQUIRE_EQUAL(obj->test, obj2->test);
        BOOST_REQUIRE_EQUAL(obj->test2, obj2->test2);
    }
}

BOOST_AUTO_TEST_CASE(serialize_manual_binary)
{
    rcc::shared_ptr<serializable_object> obj = serializable_object::Create();
    {
        std::ofstream ofs("test.bin");
        cereal::PortableBinaryOutputArchive archive(ofs);
        obj->test = 10;
        obj->test2 = 100;
        archive(*(obj.Get()));
    }
    {
        std::ifstream ifs("test.bin");
        cereal::PortableBinaryInputArchive inar(ifs);
        rcc::shared_ptr<serializable_object> obj2 = serializable_object::Create();
        inar(*(obj2.Get()));
        BOOST_REQUIRE_EQUAL(obj->test, obj2->test);
        BOOST_REQUIRE_EQUAL(obj->test2, obj2->test2);
    }
}

BOOST_AUTO_TEST_CASE(serialize_by_policy_xml)
{
    rcc::shared_ptr<serializable_object> obj = serializable_object::Create();
	obj->test = 14;
	obj->test2 = 13;
    {
        std::ofstream ofs("test2.xml");
        SerializerFactory::Serialize(obj, ofs, ISerializer::xml_e);
    }
    {
        std::ifstream ifs("test2.xml");
        SerializerFactory::DeSerialize(obj.Get(), ifs, ISerializer::xml_e);
    }
}

BOOST_AUTO_TEST_CASE(serialize_by_policy_binary)
{
	rcc::shared_ptr<serializable_object> obj = serializable_object::Create();
	obj->test = 14;
	obj->test2 = 13;
	{
		std::ofstream ofs("test2.bin", std::ios::binary);
		SerializerFactory::Serialize(obj, ofs, ISerializer::Binary_e);
	}
	{
		std::ifstream ifs("test2.bin", std::ios::binary);
		SerializerFactory::DeSerialize(obj.Get(), ifs, ISerializer::Binary_e);
		BOOST_REQUIRE_EQUAL(obj->test, 14);
		BOOST_REQUIRE_EQUAL(obj->test2, 13);
	}
}

BOOST_AUTO_TEST_CASE(deserialize_to_new_object_xml)
{
    std::ifstream ifs("test2.xml");
    auto obj = SerializerFactory::DeSerialize(ifs, ISerializer::xml_e);
    BOOST_REQUIRE(obj);
	auto typed = obj.DynamicCast<serializable_object>();
	BOOST_REQUIRE(typed);
	BOOST_REQUIRE_EQUAL(typed->test, 14);
	BOOST_REQUIRE_EQUAL(typed->test2, 13);
}

BOOST_AUTO_TEST_CASE(deserialize_to_new_object_binary)
{
	std::ifstream ifs("test2.bin", std::ios::binary);
	auto obj = SerializerFactory::DeSerialize(ifs, ISerializer::Binary_e);
	BOOST_REQUIRE(obj);
	auto typed = obj.DynamicCast<serializable_object>();
	BOOST_REQUIRE(typed);
    BOOST_REQUIRE_EQUAL(typed->test, 14);
	BOOST_REQUIRE_EQUAL(typed->test2, 13);
}

BOOST_AUTO_TEST_CASE(serialize_multi_by_policy_binary)
{
	rcc::shared_ptr<serializable_object> obj1 = serializable_object::Create();
	rcc::shared_ptr<serializable_object> obj2 = serializable_object::Create();
	obj1->test = 14;
	obj1->test2 = 13;
	obj2->test = 15;
	obj2->test2 = 16;

	{
		std::ofstream ofs("test2.bin", std::ios::binary);
		SerializerFactory::Serialize(obj1.Get(), ofs, ISerializer::Binary_e);
		SerializerFactory::Serialize(obj2.Get(), ofs, ISerializer::Binary_e);
	}
	{
		std::ifstream ifs("test2.bin", std::ios::binary);
		auto new_obj1 = rcc::shared_ptr<serializable_object>(SerializerFactory::DeSerialize(ifs, ISerializer::Binary_e));
		auto new_obj2 = rcc::shared_ptr<serializable_object>(SerializerFactory::DeSerialize(ifs, ISerializer::Binary_e));
		BOOST_REQUIRE_EQUAL(new_obj1->test, 14);
		BOOST_REQUIRE_EQUAL(new_obj1->test2, 13);
		BOOST_REQUIRE_EQUAL(new_obj2->test, 15);
		BOOST_REQUIRE_EQUAL(new_obj2->test2, 16);
	}
}



BOOST_AUTO_TEST_CASE(cleanup)
{
    delete cb;
}