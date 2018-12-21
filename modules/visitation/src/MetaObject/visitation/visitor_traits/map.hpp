#pragma once
#include "../IDynamicVisitor.hpp"
#include "string.hpp"

#include <map>
#include <utility>

namespace mo
{
    template <class K, class V>
    struct KVP
    {
        KVP() = default;
        KVP(const std::pair<const K, V>& other) : key(other.first), value(other.second) {}

        K key;
        V value;
    };

    template <class K, class V>
    struct KVP<K, V&>
    {
        KVP(std::pair<const K, V>& other) : key(other.first), value(other.second) {}

        K key;
        V& value;
    };

    template <class K, class V>
    struct KVP<K, const V&>
    {
        KVP(const std::pair<const K, V>& other) : key(other.first), value(other.second) {}

        K key;
        const V& value;
    };

    template <class T1, class T2>
    struct TTraits<KVP<T1, T2>, void> : public IStructTraits
    {
        using base = IStructTraits;

        TTraits(KVP<T1, T2>* ptr, const KVP<T1, T2>* const_ptr) : m_ptr(ptr), m_const_ptr(const_ptr) {}

        virtual void visit(IReadVisitor* visitor) override
        {
            (*visitor)(&m_ptr->key, "key");
            (*visitor)(&m_ptr->value, "value");
        }

        virtual void visit(IWriteVisitor* visitor) const override
        {
            if (m_const_ptr)
            {
                (*visitor)(&m_const_ptr->key, "key");
                (*visitor)(&m_const_ptr->value, "value");
            }
            else
            {
                (*visitor)(&m_ptr->key, "key");
                (*visitor)(&m_ptr->value, "value");
            }
        }

        virtual size_t size() const { return sizeof(KVP<T1, T2>); }
        virtual bool triviallySerializable() const { return std::is_pod<T1>::value && std::is_pod<T2>::value; }
        virtual bool isPrimitiveType() const { return false; }
        virtual TypeInfo type() const { return TypeInfo(typeid(KVP<T1, T2>)); }
        virtual const void* ptr() const { return m_ptr; }
        virtual void* ptr() { return m_ptr; }
        virtual std::string getName() const { return TypeInfo(typeid(KVP<T1, T2>)).name(); }

      private:
        KVP<T1, T2>* m_ptr;
        const KVP<T1, T2>* m_const_ptr;
    };

    template <class T1, class T2>
    struct TTraits<KVP<T1, const T2&>, void> : public IStructTraits
    {
        using base = IStructTraits;

        TTraits(KVP<T1, const T2&>* ptr, const KVP<T1, const T2&>* const_ptr) : m_ptr(ptr), m_const_ptr(const_ptr) {}

        virtual void visit(IReadVisitor* ) override
        {

        }

        virtual void visit(IWriteVisitor* visitor) const override
        {
            if (m_const_ptr)
            {
                (*visitor)(&m_const_ptr->key, "key");
                (*visitor)(&m_const_ptr->value, "value");
            }
            else
            {
                (*visitor)(&m_ptr->key, "key");
                (*visitor)(&m_ptr->value, "value");
            }
        }

        virtual size_t size() const { return sizeof(KVP<T1, T2>); }
        virtual bool triviallySerializable() const { return std::is_pod<T1>::value && std::is_pod<T2>::value; }
        virtual bool isPrimitiveType() const { return false; }
        virtual TypeInfo type() const { return TypeInfo(typeid(KVP<T1, T2>)); }
        virtual const void* ptr() const { return m_ptr; }
        virtual void* ptr() { return m_ptr; }
        virtual std::string getName() const { return TypeInfo(typeid(KVP<T1, T2>)).name(); }

      private:
        KVP<T1, const T2&>* m_ptr;
        const KVP<T1, const T2&>* m_const_ptr;
    };

    template <class K, class V>
    struct TTraits<std::map<K, V>, void> : public IContainerTraits
    {
        using base = IContainerTraits;

        TTraits(std::map<K, V>* ptr, const std::map<K, V>* const_ptr) : m_ptr(ptr), m_const_ptr(const_ptr) {}

        virtual void visit(IReadVisitor* visitor_) override
        {
            auto& visitor = *visitor_;
            for (size_t i = 0; i < num_to_read; ++i)
            {
                KVP<K, V> kvp;
                visitor(&kvp);

                (*m_ptr)[std::move(kvp.key)] = std::move(kvp.value);
            }
        }

        virtual void visit(IWriteVisitor* visitor_) const override
        {
            auto& visitor = *visitor_;
            if (m_const_ptr)
            {
                for (auto itr = m_const_ptr->begin(); itr != m_const_ptr->end(); ++itr)
                {
                    KVP<K, const V&> pair(*itr);
                    visitor(&pair);
                }
            }
            else
            {
                for (auto itr = m_ptr->begin(); itr != m_ptr->end(); ++itr)
                {
                    KVP<K, V&> pair(*itr);
                    visitor(&pair);
                }
            }
        }

        virtual TypeInfo keyType() const override { return TypeInfo(typeid(K)); }
        virtual TypeInfo valueType() const override { return TypeInfo(typeid(V)); }
        virtual TypeInfo type() const { return TypeInfo(typeid(std::map<K, V>)); }
        virtual bool isContinuous() const override { return false; }
        virtual bool podValues() const override { return std::is_pod<V>::value; }
        virtual bool podKeys() const override { return std::is_pod<K>::value; }
        virtual size_t getSize() const override { return (m_ptr != nullptr) ? m_ptr->size() : m_const_ptr->size(); }
        virtual void setSize(const size_t num) override { num_to_read = num; }
        virtual std::string getName() const { return TypeInfo(typeid(std::map<K, V>)).name(); }
      private:
        std::map<K, V>* m_ptr;
        const std::map<K, V>* m_const_ptr;
        size_t num_to_read = 0;
    };

    template <class V>
    struct TTraits<std::map<std::string, V>, void> : public IContainerTraits
    {
        using base = IContainerTraits;

        TTraits(std::map<std::string, V>* ptr, const std::map<std::string, V>* const_ptr)
            : m_ptr(ptr), m_const_ptr(const_ptr)
        {
        }

        virtual void visit(IReadVisitor* visitor_) override
        {
            auto& visitor = *visitor_;
            const auto trait = visitor.traits();

            for (size_t i = 0; i < num_to_read; ++i)
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

                (*m_ptr)[std::move(key)] = std::move(val);
            }
        }

        virtual void visit(IWriteVisitor* visitor_) const override
        {

            auto& visitor = *visitor_;
            const auto trait = visitor.traits();
            if (m_const_ptr)
            {
                for (auto itr = m_const_ptr->begin(); itr != m_const_ptr->end(); ++itr)
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
            else
            {
                for (auto itr = m_ptr->begin(); itr != m_ptr->end(); ++itr)
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
        }

        virtual TypeInfo keyType() const override { return TypeInfo(typeid(std::string)); }
        virtual TypeInfo valueType() const override { return TypeInfo(typeid(V)); }
        virtual TypeInfo type() const { return TypeInfo(typeid(std::map<std::string, V>)); }
        virtual bool isContinuous() const override { return false; }
        virtual bool podValues() const override { return std::is_pod<V>::value; }
        virtual bool podKeys() const override { return std::is_pod<std::string>::value; }
        virtual size_t getSize() const override { return (m_ptr != nullptr) ? m_ptr->size() : m_const_ptr->size(); }
        virtual void setSize(const size_t num) override { num_to_read = num; }
        virtual std::string getName() const { return TypeInfo(typeid(std::map<std::string, V>)).name(); }
      private:
        std::map<std::string, V>* m_ptr;
        const std::map<std::string, V>* m_const_ptr;
        size_t num_to_read = 0;
    };
}
