#include "TestObjects.hpp"
#include <MetaObject/object/test/object_dynamic_reflection.hpp>

#include <gtest/gtest.h>
using namespace test;

TEST(object_initialization, SerializableObject)
{
    auto stream = IAsyncStream::create();
    {
        auto inst = mo::MetaObjectFactory::instance()->create("SerializableObject");
        ASSERT_NE(inst, nullptr) << "Unable to create a SerializableObject, available objects: "
                                 << mo::MetaObjectFactory::instance()->listConstructableObjects();
        rcc::shared_ptr<SerializableObject> casted(inst);
        ASSERT_NE(casted, nullptr);
    }
    {
        auto inst = SerializableObject::create();
        ASSERT_NE(inst, nullptr);
    }
}

template <class SIG, class T>
void checkSlot(rcc::shared_ptr<T> inst, const std::string& name)
{
    auto slot = inst->getSlot(name, mo::TypeInfo(typeid(SIG)));
    ASSERT_NE(slot, nullptr);
    ASSERT_EQ(slot->getSignature(), mo::TypeInfo(typeid(SIG)));
}

template <class SIG, class T>
void checkSignal(rcc::shared_ptr<T> inst, const std::string& name)
{
    auto sig = inst->getSignal(name, mo::TypeInfo(typeid(SIG)));
    ASSERT_NE(sig, nullptr);
    ASSERT_EQ(sig->getSignature(), mo::TypeInfo(typeid(SIG)));
}

template <class T, class U>
void checkParam(rcc::shared_ptr<U> inst, const std::string& name, const T& val)
{
    auto param = inst->getParam(name);
    ASSERT_NE(param, nullptr);
    ASSERT_EQ(param->getTypeInfo(), mo::TypeInfo(typeid(T)));
    auto tdata = param->template getTypedData<T>();
    ASSERT_TRUE(tdata);
    ASSERT_EQ(tdata->data, val);
    T tmp;
    ASSERT_TRUE(param->getTypedData(&tmp));
    ASSERT_EQ(tmp, val);
}

TEST(object_initialization, SerializableObject_)
{
    auto stream = IAsyncStream::create();
    auto inst = SerializableObject::create();
    testDynamicReflection(inst);
}

TEST(object_initialization, DerivedParams_)
{
    auto stream = IAsyncStream::create();
    auto inst = DerivedParams::create();
    testDynamicReflection(inst);
    inst->base_param = 10;
    inst->derived_param = 100;
    inst->initParams(true);
    testDynamicReflection(inst);
}

TEST(object_initialization, DerivedSignals_)
{
    auto stream = IAsyncStream::create();
    auto inst = DerivedSignals::create();
    testDynamicReflection(inst);
}

TEST(object_initialization, MultipleInheritance_)
{
    auto stream = IAsyncStream::create();
    auto inst = MultipleInheritance::create();
    testDynamicReflection(inst);
}

TEST(object_initialization, Derived1_)
{
    auto stream = IAsyncStream::create();
    auto inst = Derived1::create();
    testDynamicReflection(inst);
}

TEST(object_initialization, MetaObjectPublisher_)
{
    auto stream = IAsyncStream::create();
    auto inst = MetaObjectPublisher::create();
    testDynamicReflection(inst);
}

TEST(object_initialization, ParamedObject_)
{
    auto stream = IAsyncStream::create();
    auto inst = ParamedObject::create();
    testDynamicReflection(inst);
}

TEST(object_initialization, MetaObjectSubscriber_)
{
    auto stream = IAsyncStream::create();
    auto inst = MetaObjectSubscriber::create();
    testDynamicReflection(inst);
}

TEST(object_initialization, MetaObjectSignals_)
{
    auto stream = IAsyncStream::create();
    auto inst = MetaObjectSubscriber::create();
    testDynamicReflection(inst);
}

TEST(object_initialization, MetaObjectSlots_)
{
    auto stream = IAsyncStream::create();
    auto inst = MetaObjectSubscriber::create();
    testDynamicReflection(inst);
}

TEST(object_initialization, MetaObjectCallback_)
{
    auto stream = IAsyncStream::create();
    auto inst = MetaObjectSubscriber::create();
    testDynamicReflection(inst);
}
