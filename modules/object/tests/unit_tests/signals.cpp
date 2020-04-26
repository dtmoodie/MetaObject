#include "TestObjects.hpp"
#include <MetaObject/object/RelayManager.hpp>

#include <gtest/gtest.h>

using namespace test;

template <class SIG>
void testGetSignal(const std::string& name, IMetaObject* obj)
{
    auto signal = obj->getSignal(name, TypeInfo::create<SIG>());
    ASSERT_NE(signal, nullptr) << "Unable to retrieve signal with name '" << name << "' and signature ["
                               << TypeInfo::create<SIG>().name() << "] from object of type " << obj->GetTypeName()
                               << "Available signals:\n"
                               << obj->getSignals();
    ASSERT_NE(dynamic_cast<TSignal<SIG>*>(signal), nullptr);
}

TEST(signal, init_and_access)
{
    auto stream = IAsyncStream::create();
    auto obj = DerivedSignals::create();
    ASSERT_NE(obj, nullptr);
    testGetSignal<void(int)>("base_signal", obj.get());
    testGetSignal<void(int)>("derived_signal", obj.get());
    testGetSignal<void(const IMetaObject&, const IParam&)>("param_added", obj.get());
    testGetSignal<void(const IMetaObject&, mo::Header, const IParam&)>("param_updated", obj.get());

    auto all_sigs = obj->getSignals();
    ASSERT_EQ(all_sigs.size(), 4);
}

TEST(signal, reception)
{
    auto stream = IAsyncStream::create();
    int value = 0;
    mo::TSlot<void(int)> m_slot([&value](int val) { value = val; });
    auto obj = DerivedSignals::create();
    ASSERT_NE(obj, nullptr);

    std::shared_ptr<RelayManager> mgr = std::make_shared<RelayManager>();
    ASSERT_EQ(obj->setupSignals(mgr), 8);
    auto connection = mgr->connect(&m_slot, "base_signal", nullptr);
    ASSERT_NE(connection, nullptr);

    obj->sig_base_signal(10);
    ASSERT_EQ(value, 10);
}
