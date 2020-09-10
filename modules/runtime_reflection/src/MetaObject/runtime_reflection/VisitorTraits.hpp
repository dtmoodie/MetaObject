#ifndef MO_VISITATION_VISITORTRAITS_HPP
#define MO_VISITATION_VISITORTRAITS_HPP
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
        void load(ILoadVisitor& visitor, void* instance, const std::string& name, size_t) const override
        {
            auto ptr = static_cast<T*>(instance);
            std::stringstream ss;
            ss << *ptr;
            auto str = ss.str();
            visitor(&str, name);
            *ptr = ct::fromString<T>(str);
        }

        void save(ISaveVisitor& visitor, const void* instance, const std::string& name, size_t) const override
        {
            auto ptr = static_cast<const T*>(instance);
            std::stringstream ss;
            ss << *ptr;
            auto str = ss.str();
            visitor(&str, name);
        }

        void visit(StaticVisitor& visitor, const std::string&) const override
        {
            const auto idx = ct::Reflect<T>::end();
            visitHelper<T>(visitor, idx);
        }

        std::string name() const override
        {
            return ct::Reflect<T>::getTypeName();
        }

        uint32_t getNumMembers() const override
        {
            return 1;
        }

        bool getMember(
            void* inst, void** member, const IStructTraits** trait, uint32_t idx, std::string* name) const override
        {
            if (idx == 0)
            {
                *member = inst;
                *trait = this;
                if (name)
                {
                    *name = "value";
                }
                return true;
            }
            return false;
        }

        bool getMember(const void* inst,
                       const void** member,
                       const IStructTraits** trait,
                       uint32_t idx,
                       std::string* name) const override
        {
            if (idx == 0)
            {
                *member = inst;
                *trait = this;
                if (name)
                {
                    *name = "value";
                }
                return true;
            }
            return false;
        }
    };

    DEFINE_HAS_MEMBER_FUNCTION(HasMemberLoad, load, void, ILoadVisitor&, const std::string&);
    DEFINE_HAS_MEMBER_FUNCTION(HasMemberSave, save, void, ISaveVisitor&, const std::string&);

    template <class T>
    struct TTraits<T, 7, ct::EnableIf<HasMemberLoad<T>::value && HasMemberSave<T>::value>> : StructBase<T>
    {

        void load(ILoadVisitor& visitor, void* instance, const std::string& name, size_t) const override
        {
            auto& ref = this->ref(instance);
            ref.load(visitor, name);
        }

        void save(ISaveVisitor& visitor, const void* instance, const std::string& name, size_t) const override
        {
            const auto& ref = this->ref(instance);
            ref.save(visitor, name);
        }

        void visit(StaticVisitor& visitor, const std::string&) const override
        {
            const auto idx = ct::Reflect<T>::end();
            visitHelper<T>(visitor, idx);
        }

        std::string name() const override
        {
            return ct::Reflect<T>::getTypeName();
        }

        // TODO another approach?
        uint32_t getNumMembers() const override
        {
            return 0;
        }

        bool getMember(
            void* inst, void** member, const IStructTraits** trait, uint32_t idx, std::string* name) const override
        {
            return false;
        }

        bool getMember(const void* inst,
                       const void** member,
                       const IStructTraits** trait,
                       uint32_t idx,
                       std::string* name) const override
        {
            return false;
        }
    };

} // namespace mo

#endif // MO_VISITATION_VISITORTRAITS_HPP
