#include "Objects.hpp"
#include <MetaObject/object/detail/IMetaObjectImpl.hpp>

#include <ct/static_asserts.hpp>

namespace test
{
    void staticChecks()
    {
        ct::StaticEquality<uint32_t, test::MetaObjectSlots::NUM_FIELDS, 2>{};
        ct::StaticEquality<uint32_t, ct::Reflect<test::MetaObjectSlots>::Bases_t::NUM_FIELDS, 0>{};
        ct::StaticEquality<uint32_t, ct::Reflect<test::MetaObjectSlots>::START_INDEX, 0>{};
        ct::StaticEquality<uint32_t, test::MetaObjectSlots::REFLECTION_COUNT, 2>{};
        ct::StaticEquality<uint32_t, ct::Reflect<test::MetaObjectSlots>::END_INDEX, 2>{};

        auto ptr0 = test::MetaObjectSlots::getPtr(ct::Indexer<0>{});
        auto ptr1 = test::MetaObjectSlots::getPtr(ct::Indexer<1>{});

        ct::StaticEquality<uint32_t, test::Base::NUM_FIELDS, 6>{};

        ct::StaticEquality<uint32_t, ct::Reflect<test::Base>::NUM_FIELDS, 6>{};
        ct::StaticEquality<uint32_t, ct::Reflect<test::Base>::START_INDEX, 0>{};
        ct::StaticEquality<uint32_t, ct::Reflect<test::Base>::END_INDEX, 6>{};

        ct::StaticEquality<uint32_t, test::DerivedParams::NUM_FIELDS, 2>{};

        ct::StaticEquality<uint32_t, ct::Reflect<test::DerivedParams>::NUM_FIELDS, 8>{};
        ct::StaticEquality<uint32_t, ct::Reflect<test::DerivedParams>::START_INDEX, 6>{};
        ct::StaticEquality<uint32_t, ct::Reflect<test::DerivedParams>::END_INDEX, 8>{};
    }

    SerializableObject::~SerializableObject()
    {
    }

    void Base::on_base_param_modified(mo::IParam*, mo::Header, mo::UpdateFlags)
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

    void MetaObjectSubscriber::onParamUpdate(IParam*, Header, UpdateFlags)
    {
        ++update_count;
    }

    void MetaObjectPublisher::onParamUpdate(IParam*, Header, UpdateFlags)
    {
        ++update_count;
    }
    void setupPlugin(SystemTable* table)
    {
        MetaObjectFactory::instance(table)->setupObjectConstructors(PerModuleInterface::GetInstance());
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
