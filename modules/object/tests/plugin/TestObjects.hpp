#ifndef MO_OBJECT_TESTS_OBJECTS_HPP
#define MO_OBJECT_TESTS_OBJECTS_HPP

#include <mo_objectplugin_export.hpp>

#include <MetaObject/object.hpp>

namespace test
{
    using namespace mo;

    struct mo_objectplugin_EXPORT SerializableObject : public MetaObject
    {
        ~SerializableObject() override;

        MO_DERIVE(SerializableObject, MetaObject)
        PARAM(int, test, 5)
        PARAM(int, test2, 6)
        MO_END;
    };

    struct mo_objectplugin_EXPORT Base : public MetaObject
    {
        MO_DERIVE(Base, MetaObject)
        PARAM(int, base_param, 5)
        PARAM_UPDATE_SLOT(base_param)
        MO_SIGNAL(void, base_signal, int)
        MO_SLOT(void, base_slot, int)
        MO_SLOT(void, override_slot, int)
        MO_END;
        virtual void foo();
        int base_count = 0;
    };

    struct mo_objectplugin_EXPORT DerivedParams : virtual public Base
    {
        MO_DERIVE(DerivedParams, Base)
        PARAM(int, derived_param, 10)
        MO_END;
    };

    struct mo_objectplugin_EXPORT DerivedSignals : virtual public Base
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
        MO_END;

        void override_slot(int value);
        int derived_count = 0;
    };

    struct mo_objectplugin_EXPORT MultipleInheritance : virtual public DerivedParams, virtual public DerivedSignals
    {
        MO_DERIVE(MultipleInheritance, DerivedParams, DerivedSignals)
        // TODO remove once we support empty classes
        PARAM(int, dummy, 0)
        MO_END;
    };

    struct mo_objectplugin_EXPORT Base1 : public TInterface<Base1, MetaObject>
    {
        MO_DERIVE(Base1, MetaObject)
        PARAM(int, dummy, 0)
        MO_END;
    };

    struct mo_objectplugin_EXPORT Derived1 : public TInterface<Derived1, Base1>
    {
        MO_DERIVE(Derived1, Base1)
        PARAM(int, dummy1, 0)
        MO_END;
    };

    struct mo_objectplugin_EXPORT MetaObjectPublisher : public MetaObject
    {
        void onParamUpdate(const IParam&, Header, UpdateFlags, IAsyncStream&) override;

        MO_DERIVE(MetaObjectPublisher, MetaObject)
        SOURCE(int, test_int, 5)
        MO_END;

        int update_count = 0;
    };

    struct mo_objectplugin_EXPORT ParamedObject : public MetaObject
    {
        MO_DERIVE(ParamedObject, MetaObject)
        PARAM(int, int_value, 0)
        PARAM(float, float_value, 0)
        PARAM(double, double_value, 0)

        INPUT(int, int_input)
        OUTPUT(int, int_output, 0)
        MO_END;
        void update(int value);
    };

    struct mo_objectplugin_EXPORT MetaObjectSubscriber : public MetaObject
    {
        MO_DERIVE(MetaObjectSubscriber, MetaObject)
        INPUT(int, test_int)
        MO_END;

        int update_count = 0;
        void onParamUpdate(const IParam&, Header, UpdateFlags, IAsyncStream&) override;
    };

    struct mo_objectplugin_EXPORT MetaObjectEmpty : public MetaObject
    {
        MO_DERIVE(MetaObjectEmpty, MetaObject)

        MO_END;
    };

    struct mo_objectplugin_EXPORT MetaObjectSignals : public MetaObject
    {
        MO_DERIVE(MetaObjectSignals, MetaObject)
        MO_SIGNAL(void, test_void)
        MO_SIGNAL(void, test_int, int)
        MO_END;
    };

    struct mo_objectplugin_EXPORT MetaObjectSlots : public MetaObject
    {
        MO_DERIVE(MetaObjectSlots, MetaObject)
        MO_SLOT(void, test_void)
        MO_SLOT(void, test_int, int)

        MO_END;

        int slot_called_count = 0;
        int slot_called_value = 0;
    };

    struct mo_objectplugin_EXPORT MetaObjectCallback : public MetaObject
    {
        MO_DERIVE(MetaObjectCallback, MetaObject)
        MO_SLOT(int, test_int)
        MO_SLOT(void, test_void)
        MO_END;
    };
    void mo_objectplugin_EXPORT setupPlugin(SystemTable* table);
} // namespace test
#endif // MO_OBJECT_TESTS_OBJECTS_HPP
