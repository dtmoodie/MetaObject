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
    using index_t = ct::index_t;
    template <index_t N>
    using Indexer = ct::Indexer<N>;

    template <class T, index_t I>
    auto visitValue(ILoadVisitor& visitor, T& obj, const Indexer<I> idx) -> ct::EnableIf<ct::IsWritable<T, I>::value>
    {
        auto accessor = ct::Reflect<T>::getPtr(idx);
        using Ret_t = decltype(accessor.set(obj));
        Ret_t tmp = accessor.set(obj);
        auto ptr = &tmp;
        const auto name = ct::getName<I, T>();
        visitor(ptr, name);
    }

    template <class T, index_t I>
    auto visitValue(ILoadVisitor&, T&, const Indexer<I>) -> ct::EnableIf<!ct::IsWritable<T, I>::value>
    {
    }

    template <class T>
    void visitHelper(ILoadVisitor& visitor, T& obj, const Indexer<0> idx)
    {
        visitValue(visitor, obj, idx);
    }

    template <class T, index_t I>
    void visitHelper(ILoadVisitor& visitor, T& obj, const Indexer<I> idx)
    {
        const auto next_index = --idx;
        visitHelper(visitor, obj, next_index);
        visitValue(visitor, obj, idx);
    }

    template <class T, index_t I>
    auto visitValue(ISaveVisitor& visitor, const T& obj, const ct::Indexer<I> idx)
        -> ct::EnableIf<ct::IsWritable<T, I>::value>
    {
        auto accessor = ct::Reflect<T>::getPtr(idx);
        using RefType = typename ct::ReferenceType<typename ct::GetType<decltype(accessor)>::type>::ConstType;
        RefType ref = static_cast<RefType>(accessor.get(obj));
        auto name = ct::getName<I, T>();
        visitor(&ref, name);
    }

    template <class T, index_t I>
    auto visitValue(ISaveVisitor&, const T&, const ct::Indexer<I>) -> ct::EnableIf<!ct::IsWritable<T, I>::value>
    {
    }

    template <class T>
    void visitHelper(ISaveVisitor& visitor, const T& obj, const Indexer<0> idx)
    {
        visitValue(visitor, obj, idx);
    }

    template <class T, index_t I>
    void visitHelper(ISaveVisitor& visitor, const T& obj, const Indexer<I> idx)
    {
        auto next_index = --idx;
        visitHelper(visitor, obj, next_index);
        visitValue(visitor, obj, idx);
    }

    template <class T, index_t I, class U = void>
    using EnableVisitation =
        ct::EnableIf<!ct::IsMemberFunction<T, I>::value &&
                         !ct::IsEnumField<decltype(ct::Reflect<T>::getPtr(ct::Indexer<I>{}))>::value,
                     U>;

    template <class T, index_t I, class U = void>
    using DisableVisitation =
        ct::EnableIf<ct::IsMemberFunction<T, I>::value ||
                         ct::IsEnumField<decltype(ct::Reflect<T>::getPtr(ct::Indexer<I>{}))>::value,
                     U>;

    template <class T, index_t I>
    auto visitValue(StaticVisitor& visitor, const Indexer<I> idx) -> EnableVisitation<T, I>
    {
        using Type = typename ct::GetType<decltype(ct::Reflect<T>::getPtr(idx))>::type;
        const auto name = ct::getName<I, T>();
        visitor.visit<typename std::decay<Type>::type>(name);
    }

    template <class T, index_t I>
    auto visitValue(StaticVisitor&, const Indexer<I>) -> DisableVisitation<T, I>
    {
    }

    template <class T>
    void visitHelper(StaticVisitor& visitor, const Indexer<0> idx)
    {
        visitValue<T>(visitor, idx);
    }

    template <class T, index_t I>
    void visitHelper(StaticVisitor& visitor, const Indexer<I> idx)
    {
        const auto next_idx = --idx;
        visitHelper<T>(visitor, next_idx);
        visitValue<T>(visitor, idx);
    }

    template <class T>
    struct TTraits<T, 4, ct::EnableIfReflected<T>> : public StructBase<T>
    {
        TTraits()
        {
        }

        void load(ILoadVisitor& visitor, void* instance, const std::string&, size_t) const override
        {
            auto ptr = static_cast<T*>(instance);
            const auto idx = ct::Reflect<T>::end();
            visitHelper(visitor, *ptr, idx);
        }

        void save(ISaveVisitor& visitor, const void* instance, const std::string&, size_t) const override
        {
            auto ptr = static_cast<const T*>(instance);
            const auto idx = ct::Reflect<T>::end();
            visitHelper(visitor, *ptr, idx);
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
    };

    template <class T>
    struct TTraits<T, 5, ct::EnableIfIsEnum<T>> : public StructBase<T>
    {
        void load(ILoadVisitor& visitor, void* instance, const std::string&, size_t) const override
        {
            auto ptr = static_cast<T*>(instance);
            std::stringstream ss;
            ss << *ptr;
            auto str = ss.str();
            visitor(&str, "value");
            *ptr = ct::fromString<T>(str);
        }

        void save(ISaveVisitor& visitor, const void* instance, const std::string&, size_t) const override
        {
            auto ptr = static_cast<const T*>(instance);
            std::stringstream ss;
            ss << *ptr;
            auto str = ss.str();
            visitor(&str, "value");
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
    };

    template <class T, class E = void>
    struct HasMemberLoad : std::false_type
    {
    };
    template <class T>
    struct HasMemberLoad<
        T,
        ct::Valid<decltype(std::declval<T>().load(std::declval<ILoadVisitor>(), std::declval<std::string>()))>>
        : std::true_type
    {
    };

    template <class T, class E = void>
    struct HasMemberSave : std::false_type
    {
    };

    template <class T>
    struct HasMemberSave<
        T,
        ct::Valid<decltype(std::declval<const T>().save(std::declval<ISaveVisitor>(), std::declval<std::string>()))>>
        : std::true_type
    {
    };

    template <class T>
    struct TTraits<T, 7, ct::EnableIf<HasMemberLoad<T>::value && HasMemberSave<T>::value>> : StructBase<T>
    {
        void load(ILoadVisitor& visitor, void* instance, const std::string&, size_t) const override
        {
            auto ref = this->ref(instance);
            ref.load(visitor);
        }

        void save(ISaveVisitor& visitor, const void* instance, const std::string&, size_t) const override
        {
            auto ref = this->ref(instance);
            ref.save(visitor);
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
    };

} // namespace mo

#endif // MO_VISITATION_VISITORTRAITS_HPP
