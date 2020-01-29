#include "TestObjects.hpp"

#include "MetaObject/object/MetaObject.hpp"
#include "MetaObject/object/detail/MetaObjectMacros.hpp"
#include "MetaObject/signals/detail/SignalMacros.hpp"
#include "MetaObject/signals/detail/SlotMacros.hpp"

#include "MetaObject/object/MetaObjectFactory.hpp"
#include "MetaObject/params/ParamInfo.hpp"
#include "MetaObject/params/ParamMacros.hpp"
#include "MetaObject/params/TInputParam.hpp"

#include <MetaObject/thread/fiber_include.hpp>

#include <gtest/gtest.h>

#include <iostream>

using namespace mo;
using namespace test;

TEST(object_reflection, compile_time_print)
{
    ct::printStructInfo<test::Base>(std::cout);
    std::cout << std::endl;

    ct::printStructInfo<test::DerivedParams>(std::cout);
}

TEST(object_reflection, print)
{
    auto info = mo::MetaObjectFactory::instance()->getObjectInfo("DerivedSignals");
    ASSERT_NE(info, nullptr);
    info->Print();
}

TEST(object_reflection, param_static)
{
    auto param_info = TMetaObjectInterfaceHelper<DerivedParams>::getParamInfoStatic();
    if (param_info.size() == 1)
    {
        if (param_info[0]->getName() == "derived_param")
        {
            std::cout << "missing base param \"base_param\"\n";
        }
        else
        {
            std::cout << "missing derived param \"derived_param\"\n";
        }
    }
    ASSERT_EQ(param_info.size(), 2);
}

TEST(object_reflection, signals_static)
{
    auto signal_info = TMetaObjectInterfaceHelper<DerivedSignals>::getSignalInfoStatic();
    ASSERT_EQ(signal_info.size(), 2);
    auto itr = std::find_if(
        signal_info.begin(), signal_info.end(), [](SignalInfo* info) { return info->name == "base_signal"; });
    ASSERT_NE(itr, signal_info.end());
    ASSERT_EQ((*itr)->signature, TypeInfo(typeid(void(int))));

    itr = std::find_if(
        signal_info.begin(), signal_info.end(), [](SignalInfo* info) { return info->name == "base_signal"; });

    ASSERT_NE(itr, signal_info.end());
    ASSERT_EQ((*itr)->signature, TypeInfo(typeid(void(int))));
}

TEST(object_reflection, slots_static)
{
    auto slot_info = TMetaObjectInterfaceHelper<DerivedSignals>::getSlotInfoStatic();
    ASSERT_EQ(slot_info.size(), 4);

    auto itr =
        std::find_if(slot_info.begin(), slot_info.end(), [](SlotInfo* info) { return info->name == "override_slot"; });
    ASSERT_NE(itr, slot_info.end());
    ASSERT_EQ((*itr)->signature, TypeInfo(typeid(void(int))));

    itr = std::find_if(slot_info.begin(), slot_info.end(), [](SlotInfo* info) { return info->name == "base_slot"; });
    ASSERT_NE(itr, slot_info.end());
    ASSERT_EQ((*itr)->signature, TypeInfo(typeid(void(int))));

    itr = std::find_if(slot_info.begin(), slot_info.end(), [](SlotInfo* info) { return info->name == "derived_slot"; });
    ASSERT_NE(itr, slot_info.end());
    ASSERT_EQ((*itr)->signature, TypeInfo(typeid(void(int))));
}

TEST(object, diamond_inheritance)
{
    // auto obj = rcc::shared_ptr<multi_derive>::create();
    auto constructor = mo::MetaObjectFactory::instance()->getConstructor("MultipleInheritance");
    ASSERT_NE(constructor, nullptr);
    auto info = constructor->GetObjectInfo();
    ASSERT_NE(info, nullptr);
    std::cout << info->Print();
    // auto meta_info = dynamic_cast<MetaObjectInfo*>(info);
    // ASSERT_TRUE(meta_info);
}
