#include "MetaObject/core/SystemTable.hpp"
#include "MetaObject/object/MetaObject.hpp"
#include "MetaObject/object/RelayManager.hpp"
#include "MetaObject/object/detail/IMetaObjectImpl.hpp"
#include "MetaObject/object/detail/MetaObjectMacros.hpp"
#include "MetaObject/params//ParamMacros.hpp"
#include "MetaObject/params/TInputParam.hpp"
#include "MetaObject/params/TParamPtr.hpp"

#include "MetaObject/signals/TSignal.hpp"
#include "MetaObject/signals/detail/SignalMacros.hpp"
#include "MetaObject/signals/detail/SlotMacros.hpp"

#include <MetaObject/runtime_reflection/visitor_traits/time.hpp>

#include "RuntimeObjectSystem/IObjectFactorySystem.h"
#include "RuntimeObjectSystem/RuntimeObjectSystem.h"
#include <MetaObject/thread/fiber_include.hpp>

#include <boost/test/auto_unit_test.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>

#include <iostream>

using namespace mo;

struct MetaObjectEmpty : public MetaObject
{
    MO_BEGIN(MetaObjectEmpty)

    MO_END;
};


struct MetaObjectSignals : public MetaObject
{
    MO_BEGIN(MetaObjectSignals)
    MO_SIGNAL(void, test_void)
    MO_SIGNAL(void, test_int, int)
    MO_END
};

struct MetaObjectSlots : public MetaObject
{
    MO_BEGIN(MetaObjectSlots)
    MO_SLOT(void, test_void)
    MO_SLOT(void, test_void, int)
    MO_END
    int slot_called = 0;
};

void MetaObjectSlots::test_void()
{
    std::cout << "test_void called\n";
    ++slot_called;
}
void MetaObjectSlots::test_void(int)
{
}

struct MetaObjectCallback : public MetaObject
{
    MO_BEGIN(MetaObjectCallback)
    MO_SLOT(int, test_int)
    MO_SLOT(void, test_void)
    MO_END
};
int MetaObjectCallback::test_int()
{
    return 5;
}
void MetaObjectCallback::test_void()
{
}

struct MetaObjectPublisher : public MetaObject
{
    MO_BEGIN(MetaObjectPublisher)
    PARAM(int, test_int, 5)
    TOOLTIP(test_int, "test tooltip")
    MO_END

    void onParamUpdate(IParam*, Header, UpdateFlags) override
    {
        ++update_count;
    }
    int update_count = 0;
};

struct Paramed_object : public MetaObject
{
    MO_BEGIN(Paramed_object);
    PARAM(int, int_value, 0);
    PARAM(float, float_value, 0);
    PARAM(double, double_value, 0);

    INPUT(int, int_input, 0);
    OUTPUT(int, int_output, 0);
    MO_END;
    void update(int value)
    {
        this->updateParam<int>("int_value", value);
    }
};

struct MetaObjectSubscriber : public MetaObject
{
    MO_BEGIN(MetaObjectSubscriber);
    INPUT(int, test_int, nullptr);
    MO_END;
    int update_count = 0;
    void onParamUpdate(IParam*, Header, UpdateFlags) override
    {
        ++update_count;
    }
};

MO_REGISTER_OBJECT(MetaObjectSignals)
MO_REGISTER_OBJECT(MetaObjectSlots)
MO_REGISTER_OBJECT(MetaObjectCallback)
MO_REGISTER_OBJECT(MetaObjectPublisher)
MO_REGISTER_OBJECT(MetaObjectSubscriber)
MO_REGISTER_OBJECT(Paramed_object)

BOOST_AUTO_TEST_CASE(access_Param)
{

    auto obj = rcc::shared_ptr<Paramed_object>::create();
    obj->getParam<int>("int_value");
    obj->getParam<double>("double_value");
    // TODO fix unit test
    // BOOST_REQUIRE_EQUAL(obj->getParamValue<int>("int_value"), 0);
    obj->update(10);
    // BOOST_REQUIRE_EQUAL(obj->getParamValue<int>("int_value"), 10);
}

BOOST_AUTO_TEST_CASE(test_meta_object_static_introspection_global)
{
    auto info = MetaObjectInfoDatabase::instance()->getMetaObjectInfo();
    BOOST_REQUIRE(info.size());
    for (auto& item : info)
    {
        std::cout << item->Print() << std::endl;
    }
}

BOOST_AUTO_TEST_CASE(test_meta_object_static_introspection_specific)
{
    auto info = MetaObjectInfoDatabase::instance()->getMetaObjectInfo("MetaObjectSignals");
    BOOST_REQUIRE(info);
    std::cout << info->Print() << std::endl;
}

BOOST_AUTO_TEST_CASE(test_meta_object_dynamic_introspection)
{
    RelayManager mgr;
    auto constructor = MetaObjectFactory::instance()->getConstructor("MetaObjectSignals");
    auto obj = constructor->Construct();
    auto state = constructor->GetState(obj->GetPerTypeId());
    auto weak_ptr = state->GetWeakPtr();
    {
        auto ptr = state->GetSharedPtr();
        rcc::shared_ptr<IMetaObject> meta_obj(ptr);
        meta_obj->setupSignals(&mgr);
        meta_obj->Init(true);
        auto signal_info = meta_obj->getSignalInfo();
        BOOST_REQUIRE(!weak_ptr.empty());
        BOOST_REQUIRE_EQUAL(signal_info.size(), 2);

        auto signals_ = meta_obj->getSignals();
        BOOST_REQUIRE_EQUAL(signals_.size(), 4);
    }
    BOOST_REQUIRE(weak_ptr.empty());
    BOOST_REQUIRE_EQUAL(constructor->GetNumberConstructedObjects(), 0);
}

BOOST_AUTO_TEST_CASE(test_meta_object_dynamic_access)
{
    RelayManager mgr;
    auto constructor = MetaObjectFactory::instance()->getConstructor("MetaObjectSignals");
    auto obj = constructor->Construct();
    auto meta_obj = dynamic_cast<IMetaObject*>(obj);
    meta_obj->Init(true);
    meta_obj->setupSignals(&mgr);

    auto signals_ = meta_obj->getSignals();
    BOOST_REQUIRE_EQUAL(signals_.size(), 4);
    int input_Param = 0;
    int call_value = 5;
    MetaObjectSignals* T = dynamic_cast<MetaObjectSignals*>(meta_obj);
    std::shared_ptr<mo::ISlot> slot(new mo::TSlot<void(int)>(
        std::bind([&input_Param](int value) { input_Param += value; }, std::placeholders::_1)));
    auto Connection = mgr.connect(slot.get(), "test_int");
    T->sig_test_int(call_value);
    BOOST_REQUIRE_EQUAL(input_Param, 5);
    T->sig_test_int(call_value);
    BOOST_REQUIRE_EQUAL(input_Param, 10);
    delete obj;
}

BOOST_AUTO_TEST_CASE(test_meta_object_external_slot)
{
    RelayManager mgr;
    auto constructor = MetaObjectFactory::instance()->getConstructor("MetaObjectSignals");
    auto obj = constructor->Construct();
    auto meta_obj = dynamic_cast<MetaObjectSignals*>(obj);
    meta_obj->setupSignals(&mgr);
    meta_obj->Init(true);
    bool slot_called = false;
    TSlot<void(int)> int_slot([&slot_called](int value) { slot_called = value == 5; });
    BOOST_REQUIRE(meta_obj->connectByName("test_int", &int_slot));
    int desired_value = 5;
    meta_obj->sig_test_int(desired_value);
    BOOST_REQUIRE(slot_called);
    delete obj;
}

BOOST_AUTO_TEST_CASE(test_meta_object_internal_slot)
{
    RelayManager mgr;
    auto constructor = MetaObjectFactory::instance()->getConstructor("MetaObjectSlots");
    BOOST_REQUIRE(constructor);
    auto obj = constructor->Construct();
    BOOST_REQUIRE(obj);
    auto meta_obj = dynamic_cast<MetaObjectSlots*>(obj);
    // auto slot = meta_obj->getSlot_test_void<void()>();
    // auto overload = meta_obj->getSlot_test_void<void(int)>();
    meta_obj->Init(true);
    meta_obj->setupSignals(&mgr);
    TSignal<void(void)> signal;
    BOOST_REQUIRE(meta_obj->connectByName("test_void", &signal));
    signal();
    BOOST_REQUIRE_EQUAL(meta_obj->slot_called, 1);
    signal();
    BOOST_REQUIRE_EQUAL(meta_obj->slot_called, 2);
    delete obj;
}
BOOST_AUTO_TEST_CASE(inter_object_T)
{
    RelayManager mgr;
    auto constructor = MetaObjectFactory::instance()->getConstructor("MetaObjectSignals");
    BOOST_REQUIRE(constructor);
    auto obj = constructor->Construct();
    BOOST_REQUIRE(obj);
    auto signal_object = dynamic_cast<MetaObjectSignals*>(obj);
    signal_object->setupSignals(&mgr);
    signal_object->Init(true);
    constructor = MetaObjectFactory::instance()->getConstructor("MetaObjectSlots");
    obj = constructor->Construct();
    auto slot_object = dynamic_cast<MetaObjectSlots*>(obj);
    slot_object->setupSignals(&mgr);
    slot_object->Init(true);

    BOOST_REQUIRE(
        IMetaObject::connect(signal_object, "test_void", slot_object, "test_void", TypeInfo(typeid(void(void)))));
    signal_object->sig_test_void();
    BOOST_REQUIRE_EQUAL(slot_object->slot_called, 1);
    delete obj;
    delete signal_object;
}

BOOST_AUTO_TEST_CASE(inter_object_named)
{
    RelayManager mgr;
    auto constructor = MetaObjectFactory::instance()->getConstructor("MetaObjectSignals");
    BOOST_REQUIRE(constructor);
    auto obj = constructor->Construct();
    BOOST_REQUIRE(obj);
    auto signal_object = dynamic_cast<MetaObjectSignals*>(obj);
    signal_object->setupSignals(&mgr);
    signal_object->Init(true);
    constructor = MetaObjectFactory::instance()->getConstructor("MetaObjectSlots");
    obj = constructor->Construct();
    auto slot_object = dynamic_cast<MetaObjectSlots*>(obj);
    slot_object->setupSignals(&mgr);
    slot_object->Init(true);

    BOOST_REQUIRE_EQUAL(IMetaObject::connect(signal_object, "test_void", slot_object, "test_void"), 1);
    signal_object->sig_test_void();
    BOOST_REQUIRE_EQUAL(slot_object->slot_called, 1);
    delete obj;
    delete signal_object;
}

BOOST_AUTO_TEST_CASE(rest)
{
    RelayManager mgr;
    {
        auto constructor = MetaObjectFactory::instance()->getConstructor("MetaObjectCallback");
        BOOST_REQUIRE(constructor);
        auto obj = constructor->Construct();
        BOOST_REQUIRE(obj);
        MetaObjectCallback* meta_obj = dynamic_cast<MetaObjectCallback*>(obj);
        meta_obj->Init(true);
        meta_obj->setupSignals(&mgr);
        TSignal<int(void)> signal;
        auto slot = meta_obj->getSlot("test_int", TypeInfo(typeid(int(void))));
        BOOST_REQUIRE(slot);
        auto Connection = slot->connect(&signal);
        BOOST_REQUIRE(Connection);
        BOOST_REQUIRE_EQUAL(signal(), 5);
        delete obj;
    }
    {
        auto constructor = MetaObjectFactory::instance()->getConstructor("MetaObjectCallback");
        BOOST_REQUIRE(constructor);
        auto obj = constructor->Construct();
        BOOST_REQUIRE(obj);
        obj->Init(true);
        MetaObjectCallback* cb = dynamic_cast<MetaObjectCallback*>(obj);
        constructor = MetaObjectFactory::instance()->getConstructor("MetaObjectSlots");
        obj = constructor->Construct();
        obj->Init(true);
        MetaObjectSlots* slot = dynamic_cast<MetaObjectSlots*>(obj);
        cb->test_void();
        delete cb;
        delete slot;
    }
    {
        auto constructor = MetaObjectFactory::instance()->getConstructor("MetaObjectPublisher");
        BOOST_REQUIRE(constructor);
        auto obj = constructor->Construct();
        BOOST_REQUIRE(obj);
        obj->Init(true);
        MetaObjectPublisher* ptr = dynamic_cast<MetaObjectPublisher*>(obj);
        BOOST_REQUIRE(ptr);
    }
}

BOOST_AUTO_TEST_CASE(test_params)
{
    RelayManager mgr;
    auto constructor = MetaObjectFactory::instance()->getConstructor("MetaObjectPublisher");
    auto obj = constructor->Construct();
    obj->Init(true);

    MetaObjectPublisher* publisher = dynamic_cast<MetaObjectPublisher*>(obj);

    BOOST_REQUIRE(publisher->getStream() == nullptr);
    auto num_signals = publisher->setupSignals(&mgr);

    BOOST_REQUIRE(publisher->getParam("test_int") != nullptr);
    auto params = publisher->getParams();
    BOOST_REQUIRE_EQUAL(params.size(), 1);

    constructor = MetaObjectFactory::instance()->getConstructor("MetaObjectSubscriber");
    obj = constructor->Construct();
    obj->Init(true);
    MetaObjectSubscriber* subscriber = dynamic_cast<MetaObjectSubscriber*>(obj);
    BOOST_REQUIRE(subscriber->getStream() == nullptr);
    subscriber->setupSignals(&mgr);

    BOOST_REQUIRE_EQUAL(publisher->update_count, 0);
    BOOST_REQUIRE_EQUAL(subscriber->update_count, 0);

    auto input_param = subscriber->getInput("test_int");
    BOOST_REQUIRE(input_param);

    auto output_param = publisher->getParam("test_int");
    BOOST_REQUIRE(output_param);

    BOOST_REQUIRE_EQUAL(subscriber->update_count, 0);
    BOOST_REQUIRE(subscriber->connectInput("test_int", publisher, output_param));
    BOOST_REQUIRE_EQUAL(subscriber->update_count, 1);
    publisher->test_int_param.updateData(10);

    BOOST_REQUIRE_EQUAL((*subscriber->test_int), 10);
    BOOST_REQUIRE_EQUAL(publisher->update_count, 1);
    BOOST_REQUIRE_EQUAL(subscriber->update_count, 2);

    delete obj;
}
