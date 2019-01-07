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

        template<class AR>
        void serialize(AR& ar)
        {
            ar(key);
            ar(value);
        }

        K key;
        V value;
    };

    template <class K, class V>
    struct KVP<K, V&>
    {
        KVP(std::pair<const K, V>& other) : key(other.first), value(other.second) {}


        template<class AR>
        void serialize(AR& ar)
        {
            ar(key);
            ar(value);
        }

        K key;
        V& value;
    };

    template <class K, class V>
    struct KVP<K, const V&>
    {
        KVP(const std::pair<const K, V>& other) : key(other.first), value(other.second) {}

        template<class AR>
        void serialize(AR& ar)
        {
            ar(key);
            ar(value);
        }

        K key;
        const V& value;
    };

    template <class T1, class T2>
    struct TTraits<KVP<T1, T2>, void> : public ILoadStructTraits
    {
        using base = ILoadStructTraits;

        TTraits(KVP<T1, T2>* ptr, const size_t count) : m_ptr(ptr), m_count(count){}

        void load(ILoadVisitor* visitor) override
        {
            (*visitor)(&m_ptr->key, "key");
            (*visitor)(&m_ptr->value, "value");
        }

        void save(ISaveVisitor* visitor) const override
        {
            (*visitor)(&m_ptr->key, "key");
            (*visitor)(&m_ptr->value, "value");
        }

        void visit(StaticVisitor* visitor) const override
        {
            visitor->template visit<T1>("key");
            visitor->template visit<T2>("value");
        }

        size_t size() const  override { return sizeof(KVP<T1, T2>); }
        bool triviallySerializable() const  override { return std::is_pod<T1>::value && std::is_pod<T2>::value; }
        bool isPrimitiveType() const  override { return false; }
        TypeInfo type() const  override { return TypeInfo(typeid(KVP<T1, T2>)); }
        const void* ptr() const  override { return m_ptr; }
        void* ptr() override { return m_ptr; }
        size_t count() const override {return m_count;}
        void increment() override {++m_ptr;}
      private:
        KVP<T1, T2>* m_ptr;
        size_t m_count;
    };

    template <class T1, class T2>
    struct TTraits<const KVP<T1, T2>, void> : public ISaveStructTraits
    {
        using base = ISaveStructTraits;

        TTraits(const KVP<T1, T2>* ptr, const size_t count) : m_ptr(ptr), m_count(count){}

        void save(ISaveVisitor* visitor) const override
        {
            (*visitor)(&m_ptr->key, "key");
            (*visitor)(&m_ptr->value, "value");
        }

        void visit(StaticVisitor* visitor) const override
        {
            visitor->template visit<T1>("key");
            visitor->template visit<T2>("value");
        }

        size_t size() const override { return sizeof(KVP<T1, T2>); }
        bool triviallySerializable() const override { return std::is_pod<T1>::value && std::is_pod<T2>::value; }
        bool isPrimitiveType() const override { return false; }
        TypeInfo type() const override { return TypeInfo(typeid(KVP<T1, T2>)); }
        const void* ptr() const override { return m_ptr; }
        void* ptr() { return nullptr; }
        size_t count() const override {return m_count;}
        void increment() override{++m_ptr;}
      private:
        const KVP<T1, T2>* m_ptr;
        size_t m_count;
    };

    template <class T1, class T2>
    struct TTraits<KVP<T1, const T2&>, void> : public ISaveStructTraits
    {
        using base = ISaveStructTraits;

        TTraits(KVP<T1, const T2&>* ptr, const size_t count) : m_ptr(ptr), m_count(count) {}

        void save(ISaveVisitor* visitor) const override
        {
            (*visitor)(&m_ptr->key, "key");
            (*visitor)(&m_ptr->value, "value");
        }

        void visit(StaticVisitor* visitor) const override
        {
            visitor->template visit<T1>("key");
            visitor->template visit<T2>("value");
        }

        size_t size() const override{ return sizeof(KVP<T1, T2>); }
        bool triviallySerializable() const override{ return std::is_pod<T1>::value && std::is_pod<T2>::value; }
        bool isPrimitiveType() const override{ return false; }
        TypeInfo type() const override{ return TypeInfo(typeid(KVP<T1, T2>)); }
        const void* ptr() const override{ return m_ptr; }
        size_t count() const override {return m_count;}
        void increment() override {++m_ptr;}

      private:
        KVP<T1, const T2&>* m_ptr;
        size_t m_count;
    };

    template <class T1, class T2>
    struct TTraits<const KVP<T1, const T2&>, void> : public ISaveStructTraits
    {
        using base = ISaveStructTraits;

        TTraits(const KVP<T1, const T2&>* ptr, const size_t count) : m_ptr(ptr), m_count(count) {}

        void save(ISaveVisitor* visitor) const override
        {
            (*visitor)(&m_ptr->key, "key");
            (*visitor)(&m_ptr->value, "value");
        }

        void visit(StaticVisitor* visitor) const override
        {
            visitor->template visit<T1>("key");
            visitor->template visit<T2>("value");
        }

        size_t size() const override { return sizeof(KVP<T1, T2>); }
        bool triviallySerializable() const override { return std::is_pod<T1>::value && std::is_pod<T2>::value; }
        bool isPrimitiveType() const override { return false; }
        TypeInfo type() const override { return TypeInfo(typeid(KVP<T1, T2>)); }
        const void* ptr() const override { return m_ptr; }
        size_t count() const override {return m_count;}
        void increment() override {++m_ptr;}
      private:
        const KVP<T1, const T2&>* m_ptr;
        size_t m_count;
    };

    template<class K, class V, bool LOAD>
    struct MapBase: public ISaveContainerTraits
    {
        using base = ISaveContainerTraits;
        TypeInfo keyType() const override { return TypeInfo(typeid(K)); }
        TypeInfo valueType() const override { return TypeInfo(typeid(V)); }
        TypeInfo type() const override{ return TypeInfo(typeid(std::map<K, V>)); }
        bool isContinuous() const override { return false; }
        bool podValues() const override { return std::is_pod<V>::value; }
        bool podKeys() const override { return std::is_pod<K>::value; }

        void visit(StaticVisitor* visitor) const override
        {
            visitor->template visit<K>("key");
            visitor->template visit<V>("value");
        }
    };

    template<class K, class V>
    struct MapBase<K, V, true>: public ILoadContainerTraits
    {
        using base = ILoadContainerTraits;
        TypeInfo keyType() const override { return TypeInfo(typeid(K)); }
        TypeInfo valueType() const override { return TypeInfo(typeid(V)); }
        TypeInfo type() const override{ return TypeInfo(typeid(std::map<K, V>)); }
        bool isContinuous() const override { return false; }
        bool podValues() const override { return std::is_pod<V>::value; }
        bool podKeys() const override { return std::is_pod<K>::value; }

        void visit(StaticVisitor* visitor) const override
        {
            visitor->template visit<K>("key");
            visitor->template visit<V>("value");
        }
    };

    template<class K, class V>
    void load(ILoadVisitor& visitor, std::map<K, V>& map, const size_t num_to_read)
    {
        for (size_t i = 0; i < num_to_read; ++i)
        {
            KVP<K, V> kvp;
            visitor(&kvp);
            map[std::move(kvp.key)] = std::move(kvp.value);
        }
    }

    template<class K, class V>
    void save(ISaveVisitor& visitor, const std::map<K, V>& map)
    {
        for (auto itr = map.begin(); itr != map.end(); ++itr)
        {
            KVP<K, const V&> pair(*itr);
            visitor(&pair);
        }
    }

    template <class K, class V>
    struct TTraits<std::map<K, V>, void> : public MapBase<K, V, true>
    {
        TTraits(std::map<K, V>* ptr) : m_ptr(ptr){}

        void load(ILoadVisitor* visitor_) override
        {
            mo::load(*visitor_, *m_ptr, num_to_read);
        }

        void save(ISaveVisitor* visitor_) const override
        {
            mo::save(*visitor_, *m_ptr);
        }

        size_t getSize() const override { return m_ptr->size();}
        void setSize(const size_t num) override { num_to_read = num; }
      private:
        std::map<K, V>* m_ptr;
        size_t num_to_read = 0;
    };

    template <class K, class V>
    struct TTraits<const std::map<K, V>, void> : public MapBase<K, V, false>
    {
        TTraits(const std::map<K, V>* ptr) : m_ptr(ptr){}

        void save(ISaveVisitor* visitor_) const override
        {
            mo::save(*visitor_, *m_ptr);
        }

        size_t getSize() const override { return m_ptr->size();}
      private:
        const std::map<K, V>* m_ptr;
        size_t num_to_read = 0;
    };

    template<class V>
    void load(ILoadVisitor& visitor, std::map<std::string, V>& map, const size_t num_to_read)
    {
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

            map[std::move(key)] = std::move(val);
        }
    }

    template<class V>
    void save(ISaveVisitor& visitor, const std::map<std::string, V>& map)
    {
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

    template <class V>
    struct TTraits<std::map<std::string, V>, void>  : virtual public MapBase<std::string, V, true>
    {
        TTraits(std::map<std::string, V>* ptr)
            : m_ptr(ptr)
        {
        }

        void load(ILoadVisitor* visitor_) override
        {
            mo::load(*visitor_, *m_ptr, num_to_read);
        }

        void save(ISaveVisitor* visitor_) const override
        {
            mo::save(*visitor_, *m_ptr);

        }

        size_t getSize() const override { return m_ptr->size();}

        void setSize(const size_t num) override { num_to_read = num; }

      private:
        std::map<std::string, V>* m_ptr;
        size_t num_to_read = 0;
    };

    template <class V>
    struct TTraits<const std::map<std::string, V>, void>  : virtual public MapBase<std::string, V, false>
    {
        TTraits(const std::map<std::string, V>* ptr)
            : m_ptr(ptr)
        {
        }

        void save(ISaveVisitor* visitor_) const override
        {
            mo::save(*visitor_, *m_ptr);
        }

        size_t getSize() const override { return m_ptr->size();}

      private:
        const std::map<std::string, V>* m_ptr;
    };
}
