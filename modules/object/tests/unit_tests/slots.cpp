#include "TestObjects.hpp"
#include <MetaObject/object/RelayManager.hpp>
#include <MetaObject/object/detail/IMetaObjectImpl.hpp>

#include <gtest/gtest.h>

using namespace test;

template <class SIG>
void testGetSlot(const std::string& name, IMetaObject* obj)
{
    auto slot = obj->getSlot(name, TypeInfo::create<SIG>());
    ASSERT_NE(slot, nullptr);
    ASSERT_NE(dynamic_cast<TSlot<SIG>*>(slot), nullptr);
    auto tslot = obj->template getSlot<SIG>(name);
    ASSERT_NE(tslot, nullptr);
    ASSERT_EQ(tslot, slot);
}

TEST(slot, init_and_get)
{
    auto stream = IAsyncStream::create();
    auto obj = DerivedSignals::create();
    ASSERT_NE(obj, nullptr);
    auto all_slots = obj->getSlots();

    testGetSlot<void(int)>("derived_slot", obj.get());
    testGetSlot<void(int)>("base_slot", obj.get());
    testGetSlot<void(int)>("override_slot", obj.get());

    ASSERT_EQ(all_slots.size(), 4);
}

TEST(slot, reception)
{
    auto stream = IAsyncStream::create();
    auto obj = DerivedSignals::create();
    ASSERT_NE(obj, nullptr);

    ASSERT_EQ(obj->base_count, 0);

    auto slot = obj->IMetaObject::template getSlot<void(int)>("base_slot");
    ASSERT_NE(slot, nullptr);
    (*slot)(10);
    ASSERT_EQ(obj->base_count, 10);
    (*slot)(10);
    ASSERT_EQ(obj->base_count, 20);

    ASSERT_EQ(obj->derived_count, 0);
    slot = obj->IMetaObject::template getSlot<void(int)>("derived_slot");
    (*slot)(10);
    ASSERT_EQ(obj->base_count, 20);
    ASSERT_EQ(obj->derived_count, 10);
    (*slot)(10);
    ASSERT_EQ(obj->derived_count, 20);

    slot = obj->IMetaObject::template getSlot<void(int)>("override_slot");
    (*slot)(10);
    ASSERT_EQ(obj->derived_count, 50);
}

TEST(slot, call_base)
{
    auto stream = IAsyncStream::create();
    auto derived_obj = DerivedSignals::create();
    TSignal<void(int)> sig;
    derived_obj->connectByName("base_slot", sig);
    ASSERT_EQ(derived_obj->base_count, 0);
    sig(100);
    ASSERT_EQ(derived_obj->base_count, 100);
}

TEST(slot, call_derived)
{
    auto stream = IAsyncStream::create();
    auto derived_obj = DerivedSignals::create();
    TSignal<void(int)> sig;
    derived_obj->connectByName("derived_slot", sig);
    ASSERT_EQ(derived_obj->derived_count, 0);
    sig(100);
    ASSERT_EQ(derived_obj->derived_count, 100);
}

TEST(slot, call_overloaded)
{
    auto stream = IAsyncStream::create();
    auto derived_obj = DerivedSignals::create();
    TSignal<void(int)> sig;
    derived_obj->connectByName("override_slot", sig);
    ASSERT_EQ(derived_obj->derived_count, 0);
    sig(100);
    ASSERT_EQ(derived_obj->derived_count, 300);
}
