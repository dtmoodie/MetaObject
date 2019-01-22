#include "Objects.hpp"

#include "MetaObject/object/MetaObject.hpp"
#include "MetaObject/object/detail/MetaObjectMacros.hpp"
#include "MetaObject/signals/detail/SignalMacros.hpp"
#include "MetaObject/signals/detail/SlotMacros.hpp"

#include "MetaObject/object/MetaObjectFactory.hpp"
#include "MetaObject/params/ParamInfo.hpp"
#include "MetaObject/params/ParamMacros.hpp"
#include "MetaObject/params/TInputParam.hpp"

#include <MetaObject/thread/fiber_include.hpp>

#include <boost/test/auto_unit_test.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>

#include <iostream>

using namespace mo;
using namespace test;

BOOST_AUTO_TEST_CASE(object_print)
{
    auto info = mo::MetaObjectFactory::instance()->getObjectInfo("DerivedSignals");
    BOOST_REQUIRE(info);
    info->Print();
}

BOOST_AUTO_TEST_CASE(param_static)
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
    BOOST_REQUIRE_EQUAL(param_info.size(), 2);
}

BOOST_AUTO_TEST_CASE(signals_static)
{
    auto signal_info = TMetaObjectInterfaceHelper<DerivedSignals>::getSignalInfoStatic();
    BOOST_REQUIRE_EQUAL(signal_info.size(), 2);
    auto itr = std::find_if(
        signal_info.begin(), signal_info.end(), [](SignalInfo* info) { return info->name == "base_signal"; });
    BOOST_REQUIRE(itr != signal_info.end());
    BOOST_REQUIRE((*itr)->signature == TypeInfo(typeid(void(int))));

    itr = std::find_if(
        signal_info.begin(), signal_info.end(), [](SignalInfo* info) { return info->name == "base_signal"; });

    BOOST_REQUIRE(itr != signal_info.end());
    BOOST_REQUIRE((*itr)->signature == TypeInfo(typeid(void(int))));
}

BOOST_AUTO_TEST_CASE(slots_static)
{
    auto slot_info = TMetaObjectInterfaceHelper<DerivedSignals>::getSlotInfoStatic();
    BOOST_REQUIRE_EQUAL(slot_info.size(), 3);

    auto itr =
        std::find_if(slot_info.begin(), slot_info.end(), [](SlotInfo* info) { return info->name == "override_slot"; });
    BOOST_REQUIRE(itr != slot_info.end());
    BOOST_REQUIRE((*itr)->signature == TypeInfo(typeid(void(int))));

    itr = std::find_if(slot_info.begin(), slot_info.end(), [](SlotInfo* info) { return info->name == "base_slot"; });
    BOOST_REQUIRE(itr != slot_info.end());
    BOOST_REQUIRE((*itr)->signature == TypeInfo(typeid(void(int))));

    itr = std::find_if(slot_info.begin(), slot_info.end(), [](SlotInfo* info) { return info->name == "derived_slot"; });
    BOOST_REQUIRE(itr != slot_info.end());
    BOOST_REQUIRE((*itr)->signature == TypeInfo(typeid(void(int))));
}

BOOST_AUTO_TEST_CASE(diamond)
{
    // auto obj = rcc::shared_ptr<multi_derive>::create();
    auto constructor = mo::MetaObjectFactory::instance()->getConstructor("MultipleInheritance");
    BOOST_REQUIRE(constructor);
    auto info = constructor->GetObjectInfo();
    std::cout << info->Print();
    // auto meta_info = dynamic_cast<MetaObjectInfo*>(info);
    // BOOST_REQUIRE(meta_info);
}
