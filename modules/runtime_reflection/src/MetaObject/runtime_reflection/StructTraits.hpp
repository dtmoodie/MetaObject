#ifndef MO_VISITATION_STRUCT_TRAITS_HPP
#define MO_VISITATION_STRUCT_TRAITS_HPP
#include "IDynamicVisitor.hpp"
#include "TraitInterface.hpp"
#include "TraitRegistry.hpp"

#include <ct/reflect.hpp>
#include <ct/reflect/visitor.hpp>

namespace mo
{
    template <class T>
    struct StructBase : virtual IStructTraits
    {
        using base = IStructTraits;
        static constexpr const bool DEFAULT = false;
        StructBase()
        {
            //TraitRegistry::registerTrait(TypeInfo(typeid(T)), this);
        }

        size_t size() const override
        {
            return sizeof(T);
        }

        bool triviallySerializable() const override
        {
            return std::is_trivially_copyable<T>::value;
        }

        TypeInfo type() const override
        {
            return TypeInfo(typeid(T));
        }

        T* ptr(void* inst) const
        {
            return static_cast<T*>(inst);
        }

        const T* ptr(const void* inst) const
        {
            return static_cast<const T*>(inst);
        }

        T& ref(void* inst) const
        {
            return *ptr(inst);
        }

        const T& ref(const void* inst) const
        {
            return *ptr(inst);
        }
    };

    struct RuntimeVisitorParams : ct::DefaultVisitorParams
    {
        constexpr static const bool ACCUMULATE_PATH = false;

        template <class T>
        constexpr static bool visitMemberFunctions(T)
        {
            return false;
        }
        template <class T>
        constexpr static bool visitMemberFunction(T)
        {
            return false;
        }
        template <class T>
        constexpr static bool visitStaticFunctions(T)
        {
            return false;
        }
        template <class T>
        constexpr static bool visitStaticFunction(T)
        {
            return false;
        }
    };

    struct RuntimeReflectionVisitor : public ct::VisitorBase<RuntimeReflectionVisitor, RuntimeVisitorParams>
    {
        template <class T>
        void visit(const T& obj, const std::string& name, ISaveVisitor& visitor)
        {
            visitor(&obj, name);
        }

        template <class T>
        void visit(T& obj, const std::string& name, ILoadVisitor& visitor)
        {
            visitor(&obj, name);
        }
    };

    template <class T>
    struct TTraits<T, 3, ct::EnableIfReflected<T>> : public StructBase<T>
    {
        void load(ILoadVisitor& visitor, void* inst, size_t) const override
        {
            auto& ref = *static_cast<T*>(inst);
            RuntimeReflectionVisitor::visit(ref, visitor);
        }

        void save(ISaveVisitor& visitor, const void* inst, size_t) const override
        {
            const auto& ref = *static_cast<const T*>(inst);
            RuntimeReflectionVisitor::visit(ref, visitor);
        }
    };
}

#endif // MO_VISITATION_STRUCT_TRAITS_HPP
