#include "TestObjects.hpp"

#include "MetaObject/core/SystemTable.hpp"
#include "MetaObject/object/MetaObject.hpp"
#include "MetaObject/object/RelayManager.hpp"
#include "MetaObject/object/detail/IMetaObjectImpl.hpp"
#include "MetaObject/object/detail/MetaObjectMacros.hpp"
#include "MetaObject/params//ParamMacros.hpp"
#include "MetaObject/params/TParamPtr.hpp"
#include "MetaObject/params/TSubscriberPtr.hpp"
#include <MetaObject/params/buffers/IBuffer.hpp>

#include "MetaObject/signals/TSignal.hpp"
#include "MetaObject/signals/detail/SignalMacros.hpp"
#include "MetaObject/signals/detail/SlotMacros.hpp"

#include <MetaObject/runtime_reflection/visitor_traits/time.hpp>

#include "RuntimeObjectSystem/IObjectFactorySystem.h"
#include "RuntimeObjectSystem/RuntimeObjectSystem.h"
#include <MetaObject/thread/fiber_include.hpp>

#include <gtest/gtest.h>
#include <iostream>

using namespace mo;

using namespace test;

TEST(object, param_access)
{
    static_assert(std::is_same<typename ParamedObject::BaseTypes, ct::VariadicTypedef<MetaObject>>::value, "");
    auto stream = IAsyncStream::create();
    auto obj = rcc::shared_ptr<ParamedObject>::create();
    ASSERT_TRUE(obj->getParam("int_value"));
    ASSERT_TRUE(obj->getParam("double_value"));
    // TODO fix unit test
    // ASSERT_EQ(obj->getParamValue<int>("int_value"), 0);
    obj->update(10);
    // ASSERT_EQ(obj->getParamValue<int>("int_value"), 10);
}

TEST(object, static_introspection_global)
{
    auto info = MetaObjectInfoDatabase::instance()->getMetaObjectInfo();
    ASSERT_GT(info.size(), 0);
    for (auto& item : info)
    {
        std::cout << item->Print() << std::endl;
    }
}

TEST(object, static_introspection_specific)
{
    auto info = MetaObjectInfoDatabase::instance()->getMetaObjectInfo("MetaObjectSignals");
    ASSERT_NE(info, nullptr);
    std::cout << info->Print() << std::endl;
}

TEST(object, dynamic_introspection)
{
    auto mgr = std::make_shared<RelayManager>();
    auto constructor = MetaObjectFactory::instance()->getConstructor("MetaObjectSignals");
    auto obj = constructor->Construct();
    rcc::shared_ptr<IMetaObject> meta_obj(obj);
    meta_obj->setupSignals(mgr);
    meta_obj->Init(true);
    auto signal_info = meta_obj->getSignalInfo();
    ASSERT_EQ(signal_info.size(), 2);
    auto signals_ = meta_obj->getSignals();
    ASSERT_EQ(signals_.size(), 4);
}

TEST(object, dynamic_access)
{
    auto mgr = std::make_shared<RelayManager>();
    auto constructor = MetaObjectFactory::instance()->getConstructor("MetaObjectSignals");
    auto obj = constructor->Construct();
    rcc::shared_ptr<IMetaObject> meta_obj(obj);
    meta_obj->Init(true);
    meta_obj->setupSignals(mgr);

    auto signals_ = meta_obj->getSignals();
    ASSERT_EQ(signals_.size(), 4);
    int input_Param = 0;
    int call_value = 5;
    rcc::shared_ptr<MetaObjectSignals> T(meta_obj);
    std::shared_ptr<mo::ISlot> slot(new mo::TSlot<void(int)>(
        std::bind([&input_Param](int value) { input_Param += value; }, std::placeholders::_1)));
    auto connection = mgr->connect(slot.get(), "test_int");
    T->sig_test_int(call_value);
    ASSERT_EQ(input_Param, 5);
    T->sig_test_int(call_value);
    ASSERT_EQ(input_Param, 10);
}

TEST(object, external_slot)
{
    auto mgr = std::make_shared<RelayManager>();
    auto constructor = MetaObjectFactory::instance()->getConstructor("MetaObjectSignals");
    auto obj = constructor->Construct();
    rcc::shared_ptr<MetaObjectSignals> meta_obj(obj);
    meta_obj->setupSignals(mgr);
    meta_obj->Init(true);
    bool slot_called = false;
    TSlot<void(int)> int_slot([&slot_called](int value) { slot_called = value == 5; });
    ASSERT_EQ(meta_obj->connectByName("test_int", int_slot), true);
    int desired_value = 5;
    meta_obj->sig_test_int(desired_value);
    ASSERT_EQ(slot_called, true);
}

TEST(object, internal_slot)
{
    auto mgr = std::make_shared<RelayManager>();
    auto constructor = MetaObjectFactory::instance()->getConstructor("MetaObjectSlots");
    ASSERT_NE(constructor, nullptr);
    auto obj = constructor->Construct();
    ASSERT_NE(obj, nullptr);
    rcc::shared_ptr<MetaObjectSlots> meta_obj(obj);
    ASSERT_NE(meta_obj, nullptr);
    // auto slot = meta_obj->getSlot_test_void<void()>();
    // auto overload = meta_obj->getSlot_test_void<void(int)>();
    meta_obj->Init(true);
    meta_obj->setupSignals(mgr);
    TSignal<void(void)> signal;
    ASSERT_EQ(meta_obj->connectByName("test_void", signal), true);
    signal();
    ASSERT_EQ(meta_obj->slot_called_count, 1);
    signal();
    ASSERT_EQ(meta_obj->slot_called_count, 2);
}

TEST(object, inter_object)
{
    auto mgr = std::make_shared<RelayManager>();
    auto constructor = MetaObjectFactory::instance()->getConstructor("MetaObjectSignals");
    ASSERT_NE(constructor, nullptr);
    auto obj = constructor->Construct();
    ASSERT_NE(obj, nullptr);
    rcc::shared_ptr<MetaObjectSignals> signal_object(obj);
    signal_object->setupSignals(mgr);
    signal_object->Init(true);
    constructor = MetaObjectFactory::instance()->getConstructor("MetaObjectSlots");
    obj = constructor->Construct();
    rcc::shared_ptr<MetaObjectSlots> slot_object(obj);
    slot_object->setupSignals(mgr);
    slot_object->Init(true);

    ASSERT_TRUE(
        IMetaObject::connect(*signal_object, "test_void", *slot_object, "test_void", TypeInfo::create<void(void)>()));
    signal_object->sig_test_void();
    ASSERT_EQ(slot_object->slot_called_count, 1);
}

TEST(object, inter_object_named)
{
    auto mgr = std::make_shared<RelayManager>();
    auto constructor = MetaObjectFactory::instance()->getConstructor("MetaObjectSignals");
    ASSERT_NE(constructor, nullptr);
    auto obj = constructor->Construct();
    ASSERT_NE(obj, nullptr);
    rcc::shared_ptr<MetaObjectSignals> signal_object(obj);
    signal_object->setupSignals(mgr);
    signal_object->Init(true);
    constructor = MetaObjectFactory::instance()->getConstructor("MetaObjectSlots");
    ASSERT_NE(constructor, nullptr);
    obj = constructor->Construct();
    rcc::shared_ptr<MetaObjectSlots> slot_object(obj);
    ASSERT_NE(slot_object, nullptr);
    slot_object->setupSignals(mgr);
    slot_object->Init(true);

    ASSERT_EQ(IMetaObject::connect(*signal_object, "test_void", *slot_object, "test_void"), 1);
    signal_object->sig_test_void();
    ASSERT_EQ(slot_object->slot_called_count, 1);
}

TEST(object, rest)
{
    auto stream = IAsyncStream::create();
    auto mgr = std::make_shared<RelayManager>();
    {
        auto constructor = MetaObjectFactory::instance()->getConstructor("MetaObjectCallback");
        ASSERT_NE(constructor, nullptr);
        auto obj = constructor->Construct();
        ASSERT_NE(obj, nullptr);
        rcc::shared_ptr<MetaObjectCallback> meta_obj(obj);
        meta_obj->Init(true);
        meta_obj->setupSignals(mgr);
        TSignal<int(void)> signal;
        auto slot = meta_obj->getSlot("test_int", TypeInfo::create<int(void)>());
        ASSERT_NE(slot, nullptr);
        auto connection = slot->connect(signal);
        ASSERT_NE(connection, nullptr);
        ASSERT_EQ(signal(), 5);
    }
    {
        auto constructor = MetaObjectFactory::instance()->getConstructor("MetaObjectCallback");
        ASSERT_NE(constructor, nullptr);
        auto obj = constructor->Construct();
        ASSERT_NE(obj, nullptr);
        obj->Init(true);
        rcc::shared_ptr<MetaObjectCallback> cb(obj);
        constructor = MetaObjectFactory::instance()->getConstructor("MetaObjectSlots");
        obj = constructor->Construct();
        obj->Init(true);
        rcc::shared_ptr<MetaObjectSlots> slot(obj);
        cb->test_void();
    }
    {
        auto constructor = MetaObjectFactory::instance()->getConstructor("MetaObjectPublisher");
        ASSERT_NE(constructor, nullptr);
        auto obj = constructor->Construct();
        ASSERT_NE(obj, nullptr);
        obj->Init(true);
        rcc::shared_ptr<MetaObjectPublisher> ptr(obj);
        ASSERT_NE(ptr, nullptr);
    }
}

void testParams(mo::BufferFlags flags)
{
    auto stream = mo::AsyncStream::create();
    auto mgr = std::make_shared<RelayManager>();
    auto publisher = MetaObjectPublisher::create();
    ASSERT_NE(publisher, nullptr);
    ASSERT_EQ(publisher->getStream().get(), stream.get());
    auto num_signals = publisher->setupSignals(mgr);
    EXPECT_GT(num_signals, 0);

    ASSERT_NE(publisher->getOutput("test_int"), nullptr);
    auto params = publisher->getParams();

    auto subscriber = MetaObjectSubscriber::create();
    ASSERT_EQ(subscriber->getStream().get(), stream.get());
    ASSERT_EQ(subscriber->getStream().get(), publisher->getStream().get());
    subscriber->setupSignals(mgr);

    ASSERT_EQ(publisher->update_count, 0);
    ASSERT_EQ(subscriber->update_count, 0);

    auto input_param = subscriber->getInput("test_int");
    ASSERT_NE(input_param, nullptr);

    auto output_param = publisher->getOutput("test_int");
    ASSERT_NE(output_param, nullptr);

    ASSERT_EQ(subscriber->update_count, 0);
    ASSERT_TRUE(subscriber->connectInput("test_int", publisher.get(), "test_int", flags));
    ASSERT_EQ(subscriber->update_count, 1);
    publisher->test_int.publish(10);

    if (flags & ct::value(mo::BufferFlags::FORCE_BUFFERED))
    {
        auto pub = subscriber->test_int_param.getPublisher();
        ASSERT_TRUE(pub);
        ASSERT_TRUE(pub->checkFlags(mo::ParamFlags::kBUFFER));
        auto buffer = dynamic_cast<mo::buffer::IBuffer*>(pub);
        ASSERT_TRUE(buffer);
        auto buffer_pub = buffer->getPublisher();
        ASSERT_EQ(buffer_pub, &publisher->test_int);
        ASSERT_EQ(buffer->getSize(), 2);
        ASSERT_TRUE(subscriber->test_int_param.getData());
    }

    ASSERT_NE(subscriber->test_int, nullptr);
    ASSERT_EQ((*subscriber->test_int), 10);
    ASSERT_EQ(publisher->update_count, 1);
    ASSERT_EQ(subscriber->update_count, 2);
}

TEST(object, params)
{
    testParams(mo::BufferFlags::DEFAULT);
}

TEST(object, map_params)
{
    testParams(mo::BufferFlags(mo::BufferFlags::MAP_BUFFER | ct::value(mo::BufferFlags::FORCE_BUFFERED)));
}

TEST(object, stream_params)
{
    testParams(mo::BufferFlags(mo::BufferFlags::STREAM_BUFFER | ct::value(mo::BufferFlags::FORCE_BUFFERED)));
}
