#include "TestObjects.hpp"
#include <MetaObject/object/detail/IMetaObjectImpl.hpp>

#include <ct/static_asserts.hpp>

namespace test
{
    void staticChecks()
    {
        ct::StaticEquality<uint32_t, test::MetaObjectSlots::NUM_FIELDS, 2>{};
        ct::StaticEquality<uint32_t, ct::Reflect<test::MetaObjectSlots>::Bases_t::NUM_FIELDS, 1>{};
        ct::StaticEquality<uint32_t, ct::Reflect<test::MetaObjectSlots>::START_INDEX, 1>{};
        ct::StaticEquality<uint32_t, test::MetaObjectSlots::NUM_FIELDS, 2>{};
        ct::StaticEquality<uint32_t, ct::Reflect<test::MetaObjectSlots>::END_INDEX, 3>{};

        auto ptr0 = test::MetaObjectSlots::getPtr(ct::Indexer<0>{});
        auto ptr1 = test::MetaObjectSlots::getPtr(ct::Indexer<1>{});
        (void)ptr0;
        (void)ptr1;
        ct::StaticEquality<uint32_t, test::Base::NUM_FIELDS, 6>{};

        ct::StaticEquality<uint32_t, ct::Reflect<test::Base>::NUM_FIELDS, 7>{};
        ct::StaticEquality<uint32_t, ct::Reflect<test::Base>::START_INDEX, 1>{};
        ct::StaticEquality<uint32_t, ct::Reflect<test::Base>::END_INDEX, 7>{};

        ct::StaticEquality<uint32_t, test::DerivedParams::NUM_FIELDS, 2>{};

        ct::StaticEquality<uint32_t, ct::Reflect<test::DerivedParams>::NUM_FIELDS, 9>{};
        ct::StaticEquality<uint32_t, ct::Reflect<test::DerivedParams>::START_INDEX, 7>{};
        ct::StaticEquality<uint32_t, ct::Reflect<test::DerivedParams>::END_INDEX, 9>{};
    }

    SerializableObject::~SerializableObject()
    {
    }

    void Base::on_base_param_modified(const mo::IParam&, mo::Header, mo::UpdateFlags, mo::IAsyncStream*)
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

    void Base::foo()
    {
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
        ++slot_called_count;
    }

    void MetaObjectSlots::test_int(int v)
    {
        slot_called_value = v;
        ++slot_called_count;
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
        this->setParamValue(std::move(value), "int_value");
    }

    void MetaObjectSubscriber::onParamUpdate(const IParam& param, Header hdr, UpdateFlags flgs, IAsyncStream*)
    {
        ++update_count;
    }

    void MetaObjectPublisher::onParamUpdate(const IParam&, Header, UpdateFlags, IAsyncStream*)
    {
        ++update_count;
    }

    void OutputParameterizedObject::increment()
    {
        // wut, get access token, get ref to data, increment
        ++output_val;
        test_output.publish(output_val);
        // old way
        // test_output++;
    }
} // namespace test

using namespace test;
MO_REGISTER_OBJECT(Base);
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

MO_REGISTER_OBJECT(InputParameterizedObject)
MO_REGISTER_OBJECT(OutputParameterizedObject)
