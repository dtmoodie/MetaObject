
#define BOOST_TEST_MAIN
#include "MetaObject/signals/detail/SignalMacros.hpp"
#include "MetaObject/signals/detail/SlotMacros.hpp"
#include "MetaObject/object/IMetaObject.hpp"
#include "MetaObject/object/RelayManager.hpp"
#include "MetaObject/signals/TSignal.hpp"
#include "MetaObject/core/detail/Counter.hpp"
#include "MetaObject/object/detail/MetaObjectMacros.hpp"
#include "MetaObject/params//ParamMacros.hpp"
#include "MetaObject/params/TParamPtr.hpp"
#include "MetaObject/params/TInputParam.hpp"
#include "MetaObject/params/ui/Qt/POD.hpp"
#include "MetaObject/params/ui/Qt/TParamProxy.hpp"

#include "RuntimeObjectSystem/RuntimeObjectSystem.h"
#include "RuntimeObjectSystem/IObjectFactorySystem.h"

#ifdef _MSC_VER
#include <boost/test/unit_test.hpp>
#else
#define BOOST_TEST_MODULE "MetaObject"
#include <boost/test/included/unit_test.hpp>
#endif

#include <iostream>

using namespace mo;

struct test_meta_obj_empty: public IMetaObject
{
    MO_BEGIN(test_meta_obj_empty);

    MO_END;
};

struct test_meta_obj_params: public IMetaObject
{

};

struct test_meta_object_signals: public IMetaObject
{
    MO_BEGIN(test_meta_object_signals);
    MO_SIGNAL(void, test_void);
    MO_SIGNAL(void, test_int, int);
    MO_END;
};

struct test_meta_object_slots: public IMetaObject
{
    MO_BEGIN(test_meta_object_slots);
        MO_SLOT(void, test_void);
        MO_SLOT(void, test_void, int);
    MO_END;
    int slot_called = 0;
};
void test_meta_object_slots::test_void()
{
    std::cout << "test_void called\n";
    ++slot_called;
}
void test_meta_object_slots::test_void(int)
{

}

struct test_meta_object_callback: public IMetaObject
{
    MO_BEGIN(test_meta_object_callback);
        MO_SLOT(int, test_int);
        MO_SLOT(void, test_void);
    MO_END;

};
int test_meta_object_callback::test_int()
{
    return 5;
}
void test_meta_object_callback::test_void()
{

}


struct test_meta_object_Param: public IMetaObject
{
    MO_BEGIN(test_meta_object_Param);
        PARAM(int, test_int, 5);
        TOOLTIP(test_int, "test tooltip")
    MO_END;
};

struct test_meta_object_input: public IMetaObject
{
    MO_BEGIN(test_meta_object_input);
        INPUT(int, test_int, nullptr);
    MO_END;
};



MO_REGISTER_OBJECT(test_meta_object_signals)
MO_REGISTER_OBJECT(test_meta_object_slots)
MO_REGISTER_OBJECT(test_meta_object_callback)
MO_REGISTER_OBJECT(test_meta_object_Param)
MO_REGISTER_OBJECT(test_meta_object_input)

//RuntimeObjectSystem obj_sys;

BOOST_AUTO_TEST_CASE(test_meta_object_static_introspection_global)
{
    MetaObjectFactory::instance()->registerTranslationUnit();
    auto info = MetaObjectInfoDatabase::instance()->getMetaObjectInfo();
    BOOST_REQUIRE(info.size());
    for (auto& item : info)
    {
        std::cout << item->Print() << std::endl;
    }
}

BOOST_AUTO_TEST_CASE(test_meta_object_static_introspection_specific)
{
    auto info = MetaObjectInfoDatabase::instance()->getMetaObjectInfo("test_meta_object_signals");
    BOOST_REQUIRE(info);
    std::cout << info->Print() << std::endl;
}


BOOST_AUTO_TEST_CASE(test_meta_object_dynamic_introspection)
{
    RelayManager mgr;
    auto constructor = MetaObjectFactory::instance()->getConstructor("test_meta_object_signals");
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

        auto signals_= meta_obj->getSignals();
        BOOST_REQUIRE_EQUAL(signals_.size(), 4);
    }
    BOOST_REQUIRE(weak_ptr.empty());
    BOOST_REQUIRE_EQUAL(constructor->GetNumberConstructedObjects(), 0);
}

BOOST_AUTO_TEST_CASE(test_meta_object_dynamic_access)
{
    RelayManager mgr;
    auto constructor = MetaObjectFactory::instance()->getConstructor("test_meta_object_signals");
    auto obj = constructor->Construct();
    auto meta_obj = static_cast<IMetaObject*>(obj);
    meta_obj->Init(true);
    meta_obj->setupSignals(&mgr);

    auto signals_ = meta_obj->getSignals();
    BOOST_REQUIRE_EQUAL(signals_.size(), 4);
    int input_Param = 0;
    int call_value = 5;
    test_meta_object_signals* T = dynamic_cast<test_meta_object_signals*>(meta_obj);
    std::shared_ptr<mo::ISlot> slot(new mo::TSlot<void(int)>(
        std::bind([&input_Param](int value)
    {
        input_Param += value;
    }, std::placeholders::_1)));
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
    auto constructor = MetaObjectFactory::instance()->getConstructor("test_meta_object_signals");
    auto obj = constructor->Construct();
    auto meta_obj = static_cast<test_meta_object_signals*>(obj);
    meta_obj->setupSignals(&mgr);
    meta_obj->Init(true);
    bool slot_called = false;
    TSlot<void(int)> int_slot([&slot_called](int value)
    {
        slot_called = value == 5;
    });
    BOOST_REQUIRE(meta_obj->connectByName("test_int", &int_slot));
    int desired_value = 5;
    meta_obj->sig_test_int(desired_value);
    BOOST_REQUIRE(slot_called);
    delete obj;
}


BOOST_AUTO_TEST_CASE(test_meta_object_internal_slot)
{
    RelayManager mgr;
    auto constructor = MetaObjectFactory::instance()->getConstructor("test_meta_object_slots");
    auto obj = constructor->Construct();
    auto meta_obj = static_cast<test_meta_object_slots*>(obj);
    //auto slot = meta_obj->getSlot_test_void<void()>();
    //auto overload = meta_obj->getSlot_test_void<void(int)>();
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
    auto constructor =MetaObjectFactory::instance()->getConstructor("test_meta_object_signals");
    auto obj = constructor->Construct();
    auto signal_object = static_cast<test_meta_object_signals*>(obj);
    signal_object->setupSignals(&mgr);
    signal_object->Init(true);
    constructor =MetaObjectFactory::instance()->getConstructor("test_meta_object_slots");
    obj = constructor->Construct();
    auto slot_object = static_cast<test_meta_object_slots*>(obj);
    slot_object->setupSignals(&mgr);
    slot_object->Init(true);

    BOOST_REQUIRE(IMetaObject::connect(signal_object, "test_void", slot_object, "test_void", TypeInfo(typeid(void(void)))));
    signal_object->sig_test_void();
    BOOST_REQUIRE_EQUAL(slot_object->slot_called, 1);
    delete obj;
    delete signal_object;
}

BOOST_AUTO_TEST_CASE(inter_object_named)
{
    RelayManager mgr;
    auto constructor = MetaObjectFactory::instance()->getConstructor("test_meta_object_signals");
    auto obj = constructor->Construct();
    auto signal_object = static_cast<test_meta_object_signals*>(obj);
    signal_object->setupSignals(&mgr);
    signal_object->Init(true);
    constructor = MetaObjectFactory::instance()->getConstructor("test_meta_object_slots");
    obj = constructor->Construct();
    auto slot_object = static_cast<test_meta_object_slots*>(obj);
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
        auto constructor = MetaObjectFactory::instance()->getConstructor("test_meta_object_callback");
        auto obj = constructor->Construct();
        test_meta_object_callback* meta_obj = static_cast<test_meta_object_callback*>(obj);
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
        auto constructor = MetaObjectFactory::instance()->getConstructor("test_meta_object_callback");
        auto obj = constructor->Construct();
        obj->Init(true);
        test_meta_object_callback* cb = static_cast<test_meta_object_callback*>(obj);
        constructor = MetaObjectFactory::instance()->getConstructor("test_meta_object_slots");
        obj = constructor->Construct();
        obj->Init(true);
        test_meta_object_slots* slot = static_cast<test_meta_object_slots*>(obj);
        cb->test_void();
        delete cb;
        delete slot;
    }
    {
        auto constructor = MetaObjectFactory::instance()->getConstructor("test_meta_object_Param");
        auto obj = constructor->Construct();
        obj->Init(true);
        //test_meta_object_Param* ptr = static_cast<test_meta_object_Param*>(obj);
    }
}


BOOST_AUTO_TEST_CASE(test_Params)
{
    RelayManager mgr;
    auto constructor = MetaObjectFactory::instance()->getConstructor("test_meta_object_Param");
    auto obj = constructor->Construct();
    obj->Init(true);
    test_meta_object_Param* ptr = static_cast<test_meta_object_Param*>(obj);
    ptr->setupSignals(&mgr);
    constructor = MetaObjectFactory::instance()->getConstructor("test_meta_object_input");
    obj = constructor->Construct();
    obj->Init(true);
    test_meta_object_input* input = static_cast<test_meta_object_input*>(obj);
    input->setupSignals(&mgr);
    delete obj;
}
