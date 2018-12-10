#include "MetaObject/object/MetaObject.hpp"
#include "MetaObject/object/detail/MetaObjectMacros.hpp"
#include "MetaObject/signals/detail/SignalMacros.hpp"
#include "MetaObject/signals/detail/SlotMacros.hpp"

#include "MetaObject/object/MetaObjectFactory.hpp"
#include "MetaObject/params/ParamInfo.hpp"
#include "MetaObject/params/ParamMacros.hpp"
#include "MetaObject/params/TInputParam.hpp"

#include <boost/fiber/recursive_timed_mutex.hpp>

#include <boost/test/auto_unit_test.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>

#include <iostream>

using namespace mo;

struct base : public MetaObject
{

    MO_BEGIN(base)
    PARAM(int, base_param, 5);
    MO_SIGNAL(void, base_signal, int);
    MO_SLOT(void, base_slot, int);
    MO_SLOT(void, override_slot, int);
    MO_END;
    int base_count = 0;
};

struct derived_Param : virtual public base
{
    MO_DERIVE(derived_Param, base);
    PARAM(int, derived_param, 10);
    MO_END;
};

struct derived_signals : virtual public base
{
    static std::string GetDescriptionStatic()
    {
        return "test description";
    }
    static std::string GetTooltipStatic()
    {
        return "test tooltip";
    }

    MO_DERIVE(derived_signals, base);
    MO_SIGNAL(void, derived_signal, int);
    MO_SLOT(void, derived_slot, int);
    MO_END;

    void override_slot(int value);
    int derived_count = 0;
};

struct multi_derive : virtual public derived_Param, virtual public derived_signals
{
    MO_DERIVE(multi_derive, derived_Param, derived_signals)

    MO_END;
};

void base::base_slot(int value)
{
    base_count += value;
}
void base::override_slot(int value)
{
    base_count += value * 2;
}
void derived_signals::derived_slot(int value)
{
    derived_count += value;
}
void derived_signals::override_slot(int value)
{
    derived_count += 3 * value;
}

struct base1 : public TInterface<base1, MetaObject>
{
    MO_BEGIN(base1);
    MO_END;
};

struct derived1 : public TInterface<derived1, base1>
{
    MO_DERIVE(derived1, base1);
    MO_END;
};

MO_REGISTER_OBJECT(derived_signals);
MO_REGISTER_OBJECT(derived_Param);
MO_REGISTER_OBJECT(derived1);
MO_REGISTER_OBJECT(multi_derive);

BOOST_AUTO_TEST_CASE(object_print)
{
    auto info = mo::MetaObjectFactory::instance()->getObjectInfo("derived_signals");
    info->Print();
}

BOOST_AUTO_TEST_CASE(Param_static)
{
    auto param_info = TMetaObjectInterfaceHelper<derived_Param>::getParamInfoStatic();
    if (param_info.size() == 1)
    {
        if (param_info[0]->getName() == "derived_param")
        {
            std::cout << "missing base Param \"base_param\"\n";
        }
        else
        {
            std::cout << "missing derived Param \"derived_param\"\n";
        }
    }
    BOOST_REQUIRE_EQUAL(param_info.size(), 2);
}

BOOST_AUTO_TEST_CASE(signals_static)
{
    auto signal_info = TMetaObjectInterfaceHelper<derived_signals>::getSignalInfoStatic();
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
    auto slot_info = TMetaObjectInterfaceHelper<derived_signals>::getSlotInfoStatic();
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

BOOST_AUTO_TEST_CASE(Param_dynamic)
{
    auto derived_obj = derived_Param::create();
    BOOST_REQUIRE_EQUAL(derived_obj->base_param, 5);
    BOOST_REQUIRE_EQUAL(derived_obj->derived_param, 10);
    derived_obj->base_param = 10;
    derived_obj->derived_param = 100;
    derived_obj->inTParams(true);
    BOOST_REQUIRE_EQUAL(derived_obj->base_param, 5);
    BOOST_REQUIRE_EQUAL(derived_obj->derived_param, 10);
}

BOOST_AUTO_TEST_CASE(call_base_slot)
{
    auto derived_obj = derived_signals::create();
    TSignal<void(int)> sig;
    derived_obj->connectByName("base_slot", &sig);
    BOOST_REQUIRE_EQUAL(derived_obj->base_count, 0);
    sig(100);
    BOOST_REQUIRE_EQUAL(derived_obj->base_count, 100);
}

BOOST_AUTO_TEST_CASE(call_derived_slot)
{
    auto derived_obj = derived_signals::create();
    TSignal<void(int)> sig;
    derived_obj->connectByName("derived_slot", &sig);
    BOOST_REQUIRE_EQUAL(derived_obj->derived_count, 0);
    sig(100);
    BOOST_REQUIRE_EQUAL(derived_obj->derived_count, 100);
}

BOOST_AUTO_TEST_CASE(call_overloaded_slot)
{
    auto derived_obj = derived_signals::create();
    TSignal<void(int)> sig;
    derived_obj->connectByName("override_slot", &sig);
    BOOST_REQUIRE_EQUAL(derived_obj->derived_count, 0);
    sig(100);
    BOOST_REQUIRE_EQUAL(derived_obj->derived_count, 300);
}

BOOST_AUTO_TEST_CASE(diamond)
{
    // auto obj = rcc::shared_ptr<multi_derive>::create();
    auto constructor = mo::MetaObjectFactory::instance()->getConstructor("multi_derive");
    BOOST_REQUIRE(constructor);
    auto info = constructor->GetObjectInfo();
    std::cout << info->Print();
    // auto meta_info = dynamic_cast<MetaObjectInfo*>(info);
    // BOOST_REQUIRE(meta_info);
}
