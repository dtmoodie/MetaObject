#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN

#include "MetaObject/IMetaObject.hpp"
#include "MetaObject/Signals/TypedSignal.hpp"
#include "MetaObject/Detail/Counter.hpp"
#include "MetaObject/Detail/MetaObjectMacros.hpp"
#include "MetaObject/Signals/detail/SignalMacros.hpp"
#include "MetaObject/Signals/detail/SlotMacros.hpp"
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
}


MO_REGISTER_OBJECT(test_meta_object_signals)
MO_REGISTER_OBJECT(test_meta_object_slots)

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
        auto signals = meta_obj->GetAllSignals();
    }
    
    auto info = MetaObjectInfoDatabase::Instance()->GetMetaObjectInfo();
    for(auto& item : info)
    {
        std::cout << item->Print();
    }
    {
        auto constructor = obj_sys.GetObjectFactorySystem()->GetConstructor("test_meta_object_slots");
        auto obj = constructor->Construct();
        auto meta_obj = static_cast<IMetaObject*>(obj);
        meta_obj->SetupSignals(&mgr);
        auto signal_info = meta_obj->GetSignalInfo();
        auto signals = meta_obj->GetAllSignals();
    }
}