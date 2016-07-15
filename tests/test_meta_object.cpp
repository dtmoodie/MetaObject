#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN

#include "MetaObject/IMetaObject.hpp"
#include "MetaObject/Signals/TypedSignal.hpp"
#include "MetaObject/Detail/Counter.hpp"
#include "MetaObject/Detail/MetaObjectMacros.hpp"
#include "MetaObject/Signals/detail/SignalMacros.hpp"
#include "MetaObject/Signals/detail/SlotMacros.hpp"
#include "MetaObject/Signals/detail/CallbackMacros.hpp"
#include "MetaObject/Parameters//ParameterMacros.hpp"
#include "RuntimeObjectSystem.h"
#include "IObjectFactorySystem.h"


#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "MetaObject"
#include <boost/test/unit_test.hpp>
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
        SIG_SEND(test_void);
        SIG_SEND(test_int, int)
    MO_END;
};

struct test_meta_object_slots: public IMetaObject
{
    MO_BEGIN(test_meta_object_slots);
        SLOT_DEF(void, test_void);
    MO_END;
};
void test_meta_object_slots::test_void()
{
    std::cout << "test_void called\n";
}

struct test_meta_object_callback: public IMetaObject
{
    MO_BEGIN(test_meta_object_callback);
        MO_CALLBACK(int, test_int);
        MO_CALLBACK(void, test_void);
    MO_END;
    void run_callback()
    {
        std::cout << test_int();
    }
};

struct test_meta_object_parameter: public IMetaObject
{
    MO_BEGIN(test_meta_object_parameter);
        PARAM(int, test_int, 5);
    MO_END;
};


MO_REGISTER_OBJECT(test_meta_object_signals)
MO_REGISTER_OBJECT(test_meta_object_slots)
MO_REGISTER_OBJECT(test_meta_object_callback)
MO_REGISTER_OBJECT(test_meta_object_parameter)

BOOST_AUTO_TEST_CASE(test_meta_object1)
{
    RuntimeObjectSystem obj_sys;
    obj_sys.Initialise(nullptr, nullptr);
    SignalManager mgr;
    {
        auto constructor = obj_sys.GetObjectFactorySystem()->GetConstructor("test_meta_object_signals");
        auto obj = constructor->Construct();
        auto meta_obj = static_cast<IMetaObject*>(obj);
        meta_obj->SetupSignals(&mgr);
        auto signal_info = meta_obj->GetSignalInfo();
        auto signals = meta_obj->GetSignals();
        delete obj;
    }
    
    auto info = MetaObjectInfoDatabase::Instance()->GetMetaObjectInfo();
    for(auto& item : info)
    {
        std::cout << item->Print() << std::endl;
        
    }
    {
        auto constructor = obj_sys.GetObjectFactorySystem()->GetConstructor("test_meta_object_slots");
        auto obj = constructor->Construct();
        auto meta_obj = static_cast<IMetaObject*>(obj);
        meta_obj->SetupSignals(&mgr);
        auto signal_info = meta_obj->GetSignalInfo();
        auto signals = meta_obj->GetSignals();
        delete obj;
    }
    {
        auto constructor = obj_sys.GetObjectFactorySystem()->GetConstructor("test_meta_object_callback");
        auto obj = constructor->Construct();
        test_meta_object_callback* meta_obj = static_cast<test_meta_object_callback*>(obj);
        meta_obj->Init(true);
        meta_obj->SetupSignals(&mgr);
        TypedSlot<int()> test_slot([]()->int
        {
            return 5;
        });
        meta_obj->ConnectCallback(&test_slot, "test_int");
        meta_obj->run_callback();
        delete obj;
    }
    {
        auto constructor = obj_sys.GetObjectFactorySystem()->GetConstructor("test_meta_object_callback");
        auto obj = constructor->Construct();
        obj->Init(true);
        test_meta_object_callback* cb = static_cast<test_meta_object_callback*>(obj);
        constructor = obj_sys.GetObjectFactorySystem()->GetConstructor("test_meta_object_slots");
        obj = constructor->Construct();
        obj->Init(true);
        test_meta_object_slots* slot = static_cast<test_meta_object_slots*>(obj);
        cb->ConnectCallbacks("test_void", "test_void", slot);
        cb->test_void();
        delete cb;
        delete slot;
    }
    {
        auto constructor = obj_sys.GetObjectFactorySystem()->GetConstructor("test_meta_object_parameter");
        auto obj = constructor->Construct();
        obj->Init(true);
        test_meta_object_parameter* ptr = static_cast<test_meta_object_parameter*>(obj);

    }
}