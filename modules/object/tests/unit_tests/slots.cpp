#include "Objects.hpp"
#include <MetaObject/object/RelayManager.hpp>
#include <MetaObject/object/detail/IMetaObjectImpl.hpp>
#include <boost/test/auto_unit_test.hpp>
#include <boost/test/test_tools.hpp>

using namespace test;

template<class SIG>
void testGetSlot(const std::string& name, IMetaObject* obj)
{
    auto slot = obj->getSlot(name, TypeInfo(typeid(SIG)));
    BOOST_REQUIRE(slot);
    BOOST_REQUIRE(dynamic_cast<TSlot<SIG>*>(slot));
    auto tslot = obj->template getSlot<SIG>(name);
    BOOST_REQUIRE(tslot);
    BOOST_REQUIRE_EQUAL(tslot, slot);
}

BOOST_AUTO_TEST_CASE(slot_init_and_get)
{
    auto obj = DerivedSignals::create();
    BOOST_REQUIRE(obj);
    auto all_slots = obj->getSlots();

    testGetSlot<void(int)>("derived_slot", obj.get());
    testGetSlot<void(int)>("base_slot", obj.get());
    testGetSlot<void(int)>("override_slot", obj.get());

    BOOST_REQUIRE_EQUAL(all_slots.size(), 3);
}

BOOST_AUTO_TEST_CASE(slot_reception)
{
    auto obj = DerivedSignals::create();
    BOOST_REQUIRE(obj);

    BOOST_REQUIRE_EQUAL(obj->base_count, 0);

    auto slot = obj->IMetaObject::template getSlot<void(int)>("base_slot");
    BOOST_REQUIRE(slot);
    (*slot)(10);
    BOOST_REQUIRE_EQUAL(obj->base_count, 10);
    (*slot)(10);
    BOOST_REQUIRE_EQUAL(obj->base_count, 20);

    BOOST_REQUIRE_EQUAL(obj->derived_count, 0);
    slot = obj->IMetaObject::template getSlot<void(int)>("derived_slot");
    (*slot)(10);
    BOOST_REQUIRE_EQUAL(obj->base_count, 20);
    BOOST_REQUIRE_EQUAL(obj->derived_count, 10);
    (*slot)(10);
    BOOST_REQUIRE_EQUAL(obj->derived_count, 20);

    slot = obj->IMetaObject::template getSlot<void(int)>("override_slot");
    (*slot)(10);
    BOOST_REQUIRE_EQUAL(obj->derived_count, 50);

}

BOOST_AUTO_TEST_CASE(call_base_slot)
{
    auto derived_obj = DerivedSignals::create();
    TSignal<void(int)> sig;
    derived_obj->connectByName("base_slot", &sig);
    BOOST_REQUIRE_EQUAL(derived_obj->base_count, 0);
    sig(100);
    BOOST_REQUIRE_EQUAL(derived_obj->base_count, 100);
}

BOOST_AUTO_TEST_CASE(call_derived_slot)
{
    auto derived_obj = DerivedSignals::create();
    TSignal<void(int)> sig;
    derived_obj->connectByName("derived_slot", &sig);
    BOOST_REQUIRE_EQUAL(derived_obj->derived_count, 0);
    sig(100);
    BOOST_REQUIRE_EQUAL(derived_obj->derived_count, 100);
}

BOOST_AUTO_TEST_CASE(call_overloaded_slot)
{
    auto derived_obj = DerivedSignals::create();
    TSignal<void(int)> sig;
    derived_obj->connectByName("override_slot", &sig);
    BOOST_REQUIRE_EQUAL(derived_obj->derived_count, 0);
    sig(100);
    BOOST_REQUIRE_EQUAL(derived_obj->derived_count, 300);
}
