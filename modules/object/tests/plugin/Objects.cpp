#include "Objects.hpp"

namespace test
{
    SerializableObject::~SerializableObject()
    {
    }
    void Base::base_slot(int value)
    {
        base_count += value;
    }
    void Base::override_slot(int value)
    {
        base_count += value * 2;
    }
    void DerivedSignals::derived_slot(int value)
    {
        derived_count += value;
    }
    void DerivedSignals::override_slot(int value)
    {
        derived_count += 3 * value;
    }

    void MetaObjectSlots::test_void()
    {
        std::cout << "test_void called\n";
        ++slot_called;
    }
    void MetaObjectSlots::test_void(int)
    {
    }


    int MetaObjectCallback::test_int()
    {
        return 5;
    }
    void MetaObjectCallback::test_void()
    {
    }

    void ParamedObject::update(int value)
    {
        this->updateParam<int>("int_value", value);
    }

    void MetaObjectSubscriber::onParamUpdate(IParam*, Header, UpdateFlags) override
    {
        ++update_count;
    }

    void MetaObjectPublisher::onParamUpdate(IParam*, Header, UpdateFlags) override
    {
        ++update_count;
    }

}




using namespace test;
MO_REGISTER_OBJECT(SerializableObject);
MO_REGISTER_OBJECT(DerivedSignals);
MO_REGISTER_OBJECT(DerivedParams);
MO_REGISTER_OBJECT(Derived1);
MO_REGISTER_OBJECT(MultipleInheritance);

MO_REGISTER_OBJECT(MetaObjectSignals)
MO_REGISTER_OBJECT(MetaObjectSlots)
MO_REGISTER_OBJECT(MetaObjectCallback)
MO_REGISTER_OBJECT(MetaObjectPublisher)
MO_REGISTER_OBJECT(MetaObjectSubscriber)
MO_REGISTER_OBJECT(ParamedObject)
