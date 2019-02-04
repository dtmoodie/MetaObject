#pragma once
#include <MetaObject/object.hpp>

namespace test
{
    using namespace mo;

    struct SerializableObject : public MetaObject
    {
        ~SerializableObject() override;
        MO_BEGIN(SerializableObject)
           PARAM(int, test, 5)
           PARAM(int, test2, 6)
        MO_END
    };

    struct Base : public MetaObject
    {

        MO_BEGIN(Base)
            PARAM(int, base_param, 5)
            PARAM_UPDATE_SLOT(base_param)
            MO_SIGNAL(void, base_signal, int)
            MO_SLOT(void, base_slot, int)
            MO_SLOT(void, override_slot, int)
        MO_END

        int base_count = 0;
    };

    struct DerivedParams : virtual public Base
    {
        MO_DERIVE(DerivedParams, Base)
            REFLECT_INTERNAL_MEMBER(int, derived_param)
        MO_END
    };

    struct DerivedSignals : virtual public Base
    {
        static std::string GetDescriptionStatic()
        {
            return "test description";
        }
        static std::string GetTooltipStatic()
        {
            return "test tooltip";
        }

        MO_DERIVE(DerivedSignals, Base)
            MO_SIGNAL(void, derived_signal, int)
            MO_SLOT(void, derived_slot, int)
        MO_END

        void override_slot(int value);
        int derived_count = 0;
    };

    struct MultipleInheritance: virtual public DerivedParams, virtual public DerivedSignals
    {
        MO_DERIVE(MultipleInheritance, DerivedParams, DerivedSignals)
            // TODO remove once we support empty classes
            PARAM(int, dummy, 0)
        MO_END
    };

    struct Base1 : public TInterface<Base1, MetaObject>
    {
        MO_BEGIN(Base1)
            PARAM(int, dummy, 0)
        MO_END
    };

    struct Derived1 : public TInterface<Derived1, Base1>
    {
        MO_DERIVE(Derived1, Base1)
            PARAM(int, dummy, 0)
        MO_END
    };

    struct MetaObjectPublisher : public MetaObject
    {
        MO_BEGIN(MetaObjectPublisher)
            PARAM(int, test_int, 5)
            //TOOLTIP(test_int, "test tooltip")
        MO_END

        void onParamUpdate(IParam*, Header, UpdateFlags) override;
        int update_count = 0;
    };

    struct ParamedObject : public MetaObject
    {
        MO_BEGIN(ParamedObject)
        PARAM(int, int_value, 0)
        PARAM(float, float_value, 0)
        PARAM(double, double_value, 0)

        INPUT(int, int_input)
        OUTPUT(int, int_output, 0)
        MO_END
        void update(int value);
    };

    struct MetaObjectSubscriber : public MetaObject
    {
        MO_BEGIN(MetaObjectSubscriber)
            INPUT(int, test_int)
        MO_END

        int update_count = 0;
        void onParamUpdate(IParam*, Header, UpdateFlags) override;
    };

    struct MetaObjectEmpty : public MetaObject
    {
        MO_BEGIN(MetaObjectEmpty)

        MO_END
    };


    struct MetaObjectSignals : public MetaObject
    {
        MO_BEGIN(MetaObjectSignals)
            MO_SIGNAL(void, test_void)
            MO_SIGNAL(void, test_int, int)
        MO_END
    };

    struct MetaObjectSlots : public MetaObject
    {
        MO_BEGIN(MetaObjectSlots)
            MO_SLOT(void, test_void)
            MO_SLOT(void, test_void, int)
        MO_END

        int slot_called = 0;
    };

    struct MetaObjectCallback : public MetaObject
    {
        MO_BEGIN(MetaObjectCallback)
            MO_SLOT(int, test_int)
            MO_SLOT(void, test_void)
        MO_END
    };
    void setupPlugin(SystemTable* table);
}
