#ifndef MO_RUNTIME_REFLECTION_VISITOR_TRAITS_MAP_HPP
#define MO_RUNTIME_REFLECTION_VISITOR_TRAITS_MAP_HPP

#include "../StructTraits.hpp"

#include "../ContainerTraits.hpp"
#include "../IDynamicVisitor.hpp"
#include "../type_traits.hpp"
#include "string.hpp"

#include <map>
#include <utility>

namespace mo
{
    template <class K, class V>
    struct KVP
    {
        KVP() = default;
        KVP(const std::pair<const K, V>& other)
            : key(other.first)
            , value(other.second)
        {
        }

        K key;
        V value;
    };
} // namespace mo

namespace ct
{
    REFLECT_TEMPLATED_BEGIN(mo::KVP)
        PUBLIC_ACCESS(key)
        PUBLIC_ACCESS(value)
    REFLECT_END;
} // namespace ct

namespace mo
{
    template <class T1, class T2>
    struct TTraits<KVP<T1, const T2&>, 6> : StructBase<KVP<T1, const T2&>>
    {
        void load(ILoadVisitor&, void*, const std::string&, size_t) const override
        {
            THROW(warn, "Unable to load to a const KVP");
        }

        void save(ISaveVisitor& visitor, const void* inst, const std::string&, size_t) const override
        {
            auto ptr = this->ptr(inst);
            visitor(&ptr->key, "key");
            visitor(&ptr->value, "value");
        }

        void visit(StaticVisitor& visitor, const std::string&) const override
        {
            visitor.template visit<T1>("key");
            visitor.template visit<T2>("value");
        }

        bool triviallySerializable() const override
        {
            return false;
        }

        // TODO another approach?
        uint32_t getNumMembers() const override
        {
            return 2;
        }

        bool loadMember(ILoadVisitor& visitor, void* inst, uint32_t idx, std::string* name) const override
        {
            auto& ref = this->ref(inst);
            if (idx == 0)
            {
                visitor(&ref.key);
            }
            return false;
        }

        bool saveMember(ISaveVisitor& visitor, const void* inst, uint32_t idx, std::string* name) const override
        {
            auto& ref = this->ref(inst);
            if (idx == 0)
            {
                visitor(&ref.key, "key");
            }
            else if (idx == 1)
            {
                visitor(&ref.value, "value");
            }
            return false;
        }
    };

    template <class K, class V>
    struct TTraits<std::map<K, V>, 4> : ContainerBase<std::map<K, V>, V, K>
    {
        void load(ILoadVisitor& visitor, void* inst, const std::string&, size_t cnt) const override
        {
            MO_ASSERT_EQ(cnt, 1);
            auto ptr = static_cast<std::map<K, V>*>(inst);
            auto size = visitor.getCurrentContainerSize();
            static_assert(ct::IsReflected<KVP<K, V>>::value, "");
            for (size_t i = 0; i < size; ++i)
            {
                KVP<K, V> kvp;
                visitor(&kvp);
                (*ptr)[std::move(kvp.key)] = std::move(kvp.value);
            }
        }

        void save(ISaveVisitor& visitor, const void* inst, const std::string&, size_t cnt) const override
        {
            MO_ASSERT_EQ(cnt, 1);
            auto ptr = static_cast<const std::map<K, V>*>(inst);
            for (auto itr = ptr->begin(); itr != ptr->end(); ++itr)
            {
                KVP<K, const V&> pair(*itr);
                visitor(&pair);
            }
        }

        void visit(StaticVisitor& visitor, const std::string&) const override
        {
            visitor.template visit<K>("keys");
            visitor.template visit<V>("values");
        }

        bool isContinuous() const override
        {
            return false;
        }

        size_t getContainerSize(const void* inst) const override
        {
            auto p = this->ptr(inst);
            return p->size();
        }

        void setContainerSize(size_t, void*) const override
        {
        }
    };

    template <class V>
    struct TTraits<std::map<std::string, V>, 5> : virtual ContainerBase<std::map<std::string, V>, V, std::string>
    {
        void load(ILoadVisitor& visitor, void* inst, const std::string&, size_t cnt) const override
        {
            MO_ASSERT_EQ(cnt, 1);
            auto& map = *this->ptr(inst);
            const auto trait = visitor.traits();
            auto sz = visitor.getCurrentContainerSize();
            for (size_t i = 0; i < sz; ++i)
            {
                std::string key;
                V val;
                if (trait.supports_named_access)
                {
                    visitor(&val);
                    key = visitor.getCurrentElementName();
                }
                else
                {
                    visitor(&key);
                    visitor(&val);
                }

                map[std::move(key)] = std::move(val);
            }
        }

        void save(ISaveVisitor& visitor, const void* inst, const std::string&, size_t cnt) const override
        {
            MO_ASSERT_EQ(cnt, 1);
            const auto& map = *this->ptr(inst);
            const auto trait = visitor.traits();
            for (auto itr = map.begin(); itr != map.end(); ++itr)
            {
                if (trait.supports_named_access)
                {
                    visitor(&itr->second, itr->first);
                }
                else
                {
                    std::string key = itr->first;
                    visitor(&key);
                    visitor(&itr->second);
                }
            }
        }

        void visit(StaticVisitor& visitor, const std::string&) const override
        {
            visitor.template visit<std::string>("keys");
            visitor.template visit<V>("values");
        }

        bool isContinuous() const override
        {
            return false;
        }

        size_t getContainerSize(const void* inst) const override
        {
            auto p = this->ptr(inst);
            return p->size();
        }

        void setContainerSize(size_t, void*) const override
        {
        }
    };
} // namespace mo

#endif // MO_RUNTIME_REFLECTION_VISITOR_TRAITS_MAP_HPP