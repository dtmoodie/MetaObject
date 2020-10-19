#ifndef MO_VISITATION_TRAIT_INTERFACE_HPP
#define MO_VISITATION_TRAIT_INTERFACE_HPP
#include "TraitRegistry.hpp"
#include "type_traits.hpp"

#include <MetaObject/detail/TypeInfo.hpp>

#include <ct/static_asserts.hpp>
#include <ct/type_traits.hpp>

namespace mo
{
    struct IDynamicVisitor;
    struct ILoadVisitor;
    struct ISaveVisitor;
    // visit an object without creating an object
    struct StaticVisitor;

    struct MO_EXPORTS ITraits
    {
        virtual ~ITraits();
        virtual TypeInfo type() const = 0;
        virtual std::string name() const;
        virtual void visit(StaticVisitor& visitor, const std::string& name) const = 0;
        virtual void
        save(ISaveVisitor& visitor, const void* instance, const std::string& name, size_t count = 1) const = 0;
        virtual void load(ILoadVisitor& visitor, void* instance, const std::string& name, size_t count = 1) const = 0;
    };

    struct MO_EXPORTS IStructTraits : virtual ITraits
    {
        using base = IStructTraits;

        // sizeof(T)
        virtual size_t size() const = 0;

        // can be serialized via a memcpy(ptr)
        virtual bool triviallySerializable() const = 0;

        virtual uint32_t getNumMembers() const;

        virtual std::string getMemberName(uint32_t idx) const;
        virtual int32_t getMemberIndex(const std::string& name) const;

        virtual bool loadMember(ILoadVisitor& visitor, void* inst, uint32_t idx, std::string* name = nullptr) const;

        virtual bool
        saveMember(ISaveVisitor& visitor, const void* inst, uint32_t idx, std::string* name = nullptr) const;

        // Lookup a member by name
        virtual bool
        loadMember(ILoadVisitor& visitor, void* inst, const std::string& name, uint32_t* idx = nullptr) const;

        virtual bool
        saveMember(ISaveVisitor& visitor, const void* inst, const std::string& name, uint32_t* idx = nullptr) const;
    };

    struct MO_EXPORTS IContainerTraits : virtual IStructTraits
    {
        using base = IContainerTraits;

        virtual TypeInfo keyType() const = 0;
        virtual TypeInfo valueType() const = 0;

        virtual bool isContinuous() const = 0;
        virtual bool podValues() const = 0;
        virtual bool podKeys() const = 0;
        virtual size_t getContainerSize(const void* inst) const = 0;
        virtual void setContainerSize(size_t size, void* inst) const = 0;

        virtual void* valuePointer(void* inst) const = 0;
        virtual const void* valuePointer(const void* inst) const = 0;
        virtual void* keyPointer(void* inst) const = 0;
        virtual const void* keyPointer(const void* inst) const = 0;
    };

    template <class T, int P = 10, class E = void>
    struct TTraits : TTraits<T, P - 1, E>
    {
    };

    template <class T>
    struct TTraits<T, 0, void>
    {
    };

    template <class T>
    struct TraitRegisterer
    {
        TraitRegisterer()
        {
            static const TTraits<T, 10, void> s_inst;
            const auto inst = &s_inst;
            const auto type = mo::TypeInfo::template create<T>();
            TraitRegistry::registerTrait(type, inst);
        }
    };

    template <class T>
    struct TTraits<T, 10, ct::EnableIf<!IsPrimitive<T>::value>> : TTraits<T, 9, void>
    {
        static const TraitRegisterer<T> s_registerer;
        TTraits()
        {
            (void)s_registerer;
            // Traits should only ever be a pointer to a vftable, thus this check ensures you don't put stuff in them
            // that doesn't belong.
            // ct::StaticEquality<size_t, sizeof(TTraits<T>), sizeof(void*)>{};
            // msvc puts stuff in there that doesn't belong -_-
        }
    };

    template <class T>
    const TraitRegisterer<T> TTraits<T, 10, ct::EnableIf<!IsPrimitive<T>::value>>::s_registerer;
} // namespace mo

#endif // MO_VISITATION_TRAIT_INTERFACE_HPP
