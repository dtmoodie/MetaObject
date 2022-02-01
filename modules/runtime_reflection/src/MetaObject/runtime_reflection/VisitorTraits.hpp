#ifndef MO_VISITATION_VISITORTRAITS_HPP
#define MO_VISITATION_VISITORTRAITS_HPP
#include "export.hpp"

#include "StructTraits.hpp"
#include "TraitRegistry.hpp"
#include "type_traits.hpp"

#include <MetaObject/logging/logging.hpp>

#include <ct/reflect.hpp>
#include <ct/reflect/print.hpp>
#include <ct/type_traits.hpp>

#include <sstream>

namespace mo
{

    template <class T>
    struct TTraits<T, 5, ct::EnableIfIsEnum<T>> : public StructBase<T>
    {
        void load(ILoadVisitor& visitor, void* instance, const std::string& name, size_t) const override;

        void save(ISaveVisitor& visitor, const void* instance, const std::string& name, size_t) const override;

        void visit(StaticVisitor& visitor, const std::string&) const override;

        std::string name() const override;

        uint32_t getNumMembers() const override;

        bool loadMember(ILoadVisitor& visitor, void* inst, uint32_t idx, std::string* name) const override;

        bool saveMember(ISaveVisitor& visitor, const void* inst, uint32_t idx, std::string* name) const override;
    };

    DEFINE_HAS_MEMBER_FUNCTION(HasMemberLoad, load, void, ILoadVisitor&, const std::string&);
    DEFINE_HAS_MEMBER_FUNCTION(HasMemberSave, save, void, ISaveVisitor&, const std::string&);

    template <class T>
    struct TTraits<T, 7, ct::EnableIf<HasMemberLoad<T>::value && HasMemberSave<T>::value>> : StructBase<T>
    {

        void load(ILoadVisitor& visitor, void* instance, const std::string& name, size_t) const override;

        void save(ISaveVisitor& visitor, const void* instance, const std::string& name, size_t) const override;

        void visit(StaticVisitor& visitor, const std::string&) const override;

        std::string name() const override;

        // TODO another approach?
        uint32_t getNumMembers() const override;
    };

} // namespace mo

#include "IDynamicVisitor.hpp"
// implementation
namespace mo
{

    template <class T>
    void TTraits<T, 5, ct::EnableIfIsEnum<T>>::load(ILoadVisitor& visitor,
                                                    void* instance,
                                                    const std::string& name,
                                                    size_t) const
    {
        auto& ref = this->ref(instance);
        if (visitor.traits().human_readable)
        {
            std::stringstream ss;
            ss << ref;
            auto str = ss.str();
            visitor(&str, name);
            ref = ct::fromString<T>(str);
        }
        else
        {
            visitor(&ref.value, name);
        }
    }

    template <class T>
    void TTraits<T, 5, ct::EnableIfIsEnum<T>>::save(ISaveVisitor& visitor,
                                                    const void* instance,
                                                    const std::string& name,
                                                    size_t) const
    {
        const auto& ref = this->ref(instance);
        if (visitor.traits().human_readable)
        {
            const std::string str = ct::toString(ref);
            visitor(&str, name);
        }
        else
        {
            visitor(&ref.value, name);
        }
    }

    template <class T>
    void TTraits<T, 5, ct::EnableIfIsEnum<T>>::visit(StaticVisitor& visitor, const std::string&) const
    {
        using U = typename std::remove_reference<decltype(std::declval<T>().value)>::type;
        visitor.template visit<U>("value");
    }

    template <class T>
    std::string TTraits<T, 5, ct::EnableIfIsEnum<T>>::name() const
    {
        return ct::Reflect<T>::getTypeName();
    }

    template <class T>
    uint32_t TTraits<T, 5, ct::EnableIfIsEnum<T>>::getNumMembers() const
    {
        return 1;
    }

    template <class T>
    bool TTraits<T, 5, ct::EnableIfIsEnum<T>>::loadMember(ILoadVisitor& visitor,
                                                          void* inst,
                                                          uint32_t idx,
                                                          std::string* name) const
    {
        if (idx == 0)
        {
            T& ref = this->ref(inst);
            if (visitor.traits().human_readable)
            {
                std::string str;
                visitor(&str, "value");
                ref = ct::fromString<T>(str);
            }
            else
            {
                visitor(&ref.value);
            }

            if (name)
            {
                *name = "value";
            }
            return true;
        }
        return false;
    }

    template <class T>
    bool TTraits<T, 5, ct::EnableIfIsEnum<T>>::saveMember(ISaveVisitor& visitor,
                                                          const void* inst,
                                                          uint32_t idx,
                                                          std::string* name) const
    {
        if (idx == 0)
        {
            const T& ref = this->ref(inst);
            if (visitor.traits().human_readable)
            {
                std::string str = ct::toString(ref);
                visitor(&str, "value");
            }
            else
            {
                visitor(&ref.value);
            }

            if (name)
            {
                *name = "value";
            }
            return true;
        }
        return false;
    }

    template <class T>
    void TTraits<T, 7, ct::EnableIf<HasMemberLoad<T>::value && HasMemberSave<T>::value>>::load(ILoadVisitor& visitor,
                                                                                               void* instance,
                                                                                               const std::string& name,
                                                                                               size_t) const
    {
        auto& ref = this->ref(instance);
        ref.load(visitor, name);
    }

    template <class T>
    void TTraits<T, 7, ct::EnableIf<HasMemberLoad<T>::value && HasMemberSave<T>::value>>::save(ISaveVisitor& visitor,
                                                                                               const void* instance,
                                                                                               const std::string& name,
                                                                                               size_t) const
    {
        const auto& ref = this->ref(instance);
        ref.save(visitor, name);
    }

    template <class T>
    void
    TTraits<T, 7, ct::EnableIf<HasMemberLoad<T>::value && HasMemberSave<T>::value>>::visit(StaticVisitor& visitor,
                                                                                           const std::string&) const
    {
        const auto idx = ct::Reflect<T>::end();
        visitHelper<T>(visitor, idx);
    }

    template <class T>
    std::string TTraits<T, 7, ct::EnableIf<HasMemberLoad<T>::value && HasMemberSave<T>::value>>::name() const
    {
        return ct::Reflect<T>::getTypeName();
    }

    template <class T>
    uint32_t TTraits<T, 7, ct::EnableIf<HasMemberLoad<T>::value && HasMemberSave<T>::value>>::getNumMembers() const
    {
        return 0;
    }

} // namespace mo

#endif // MO_VISITATION_VISITORTRAITS_HPP
