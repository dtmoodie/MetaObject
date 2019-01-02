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

        TTraits(KVP<T1, T2>* ptr) : m_ptr(ptr){}

        void visit(IReadVisitor* visitor) override
        {
            (*visitor)(&m_ptr->key, "key");
            (*visitor)(&m_ptr->value, "value");
        }

        void visit(IWriteVisitor* visitor) const override
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

      private:
        KVP<T1, T2>* m_ptr;
    };

    template <class T1, class T2>
    struct TTraits<const KVP<T1, T2>, void> : public IStructTraits
    {
        using base = IStructTraits;

        TTraits(const KVP<T1, T2>* ptr) : m_ptr(ptr){}

        void visit(IReadVisitor* ) override
        {

        }

        void visit(IWriteVisitor* visitor) const override
        {
            (*visitor)(&m_ptr->key, "key");
            (*visitor)(&m_ptr->value, "value");
        }

        void visit(StaticVisitor* visitor) const override
        {
            visitor->template visit<T1>("key");
            visitor->template visit<T2>("value");
        }

        virtual size_t size() const { return sizeof(KVP<T1, T2>); }
        virtual bool triviallySerializable() const { return std::is_pod<T1>::value && std::is_pod<T2>::value; }
        virtual bool isPrimitiveType() const { return false; }
        virtual TypeInfo type() const { return TypeInfo(typeid(KVP<T1, T2>)); }
        virtual const void* ptr() const { return m_ptr; }
        virtual void* ptr() { return nullptr; }

      private:
        const KVP<T1, T2>* m_ptr;
    };

    template <class T1, class T2>
    struct TTraits<KVP<T1, const T2&>, void> : public IStructTraits
    {
        using base = IStructTraits;

        TTraits(KVP<T1, const T2&>* ptr) : m_ptr(ptr) {}

        void visit(IReadVisitor* ) override
        {

        }

        void visit(IWriteVisitor* visitor) const override
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
        void* ptr() override{ return m_ptr; }

      private:
        KVP<T1, const T2&>* m_ptr;
    };

    template <class T1, class T2>
    struct TTraits<const KVP<T1, const T2&>, void> : public IStructTraits
    {
        using base = IStructTraits;

        TTraits(const KVP<T1, const T2&>* ptr) : m_ptr(ptr) {}

        void visit(IReadVisitor* ) override{}

        void visit(IWriteVisitor* visitor) const override
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
        void* ptr() override { return nullptr; }

      private:
        const KVP<T1, const T2&>* m_ptr;
    };

    template<class K, class V>
    struct MapBase: public IContainerTraits
    {
        using base = IContainerTraits;
        TypeInfo keyType() const override { return TypeInfo(typeid(K)); }
        TypeInfo valueType() const override { return TypeInfo(typeid(V)); }
        TypeInfo type() const { return TypeInfo(typeid(std::map<K, V>)); }
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
    void read(IReadVisitor& visitor, std::map<K, V>& map, const size_t num_to_read)
    {
        for (size_t i = 0; i < num_to_read; ++i)
        {
            KVP<K, V> kvp;
            visitor(&kvp);
            map[std::move(kvp.key)] = std::move(kvp.value);
        }
    }

    template<class K, class V>
    void write(IWriteVisitor& visitor, const std::map<K, V>& map)
    {
        for (auto itr = map.begin(); itr != map.end(); ++itr)
        {
            KVP<K, const V&> pair(*itr);
            visitor(&pair);
        }
    }

    template <class K, class V>
    struct TTraits<std::map<K, V>, void> : public MapBase<K, V>
    {
        TTraits(std::map<K, V>* ptr) : m_ptr(ptr){}

        virtual void visit(IReadVisitor* visitor_) override
        {
            read(*visitor_, *m_ptr, num_to_read);
        }

        virtual void visit(IWriteVisitor* visitor_) const override
        {
            write(*visitor_, *m_ptr);
        }

        virtual size_t getSize() const override { return m_ptr->size();}
        virtual void setSize(const size_t num) override { num_to_read = num; }
      private:
        std::map<K, V>* m_ptr;
        size_t num_to_read = 0;
    };

    template <class K, class V>
    struct TTraits<const std::map<K, V>, void> : public MapBase<K, V>
    {
        TTraits(const std::map<K, V>* ptr) : m_ptr(ptr){}

        virtual void visit(IReadVisitor*) override
        {
            throw std::runtime_error("Trying to read into a const map");
        }

        virtual void visit(IWriteVisitor* visitor_) const override
        {
            write(*visitor_, *m_ptr);
        }

        virtual size_t getSize() const override { return m_ptr->size();}
        virtual void setSize(const size_t num) override { num_to_read = num; }
      private:
        const std::map<K, V>* m_ptr;
        size_t num_to_read = 0;
    };

    template<class V>
    void read(IReadVisitor& visitor, std::map<std::string, V>& map, const size_t num_to_read)
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
    void write(IWriteVisitor& visitor, const std::map<std::string, V>& map)
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
    struct TTraits<std::map<std::string, V>, void>  : public MapBase<std::string, V>
    {
        TTraits(std::map<std::string, V>* ptr)
            : m_ptr(ptr)
        {
        }

        virtual void visit(IReadVisitor* visitor_) override
        {
            read(*visitor_, *m_ptr, num_to_read);

        }

        virtual void visit(IWriteVisitor* visitor_) const override
        {
            write(*visitor_, *m_ptr);

        }
        virtual size_t getSize() const override { return m_ptr->size();}
        virtual void setSize(const size_t num) override { num_to_read = num; }
      private:
        std::map<std::string, V>* m_ptr;
        size_t num_to_read = 0;
    };

    template <class V>
    struct TTraits<const std::map<std::string, V>, void>  : public MapBase<std::string, V>
    {
        TTraits(const std::map<std::string, V>* ptr)
            : m_ptr(ptr)
        {
        }

        virtual void visit(IReadVisitor* ) override
        {
            throw std::runtime_error("Trying to read into a constant map");
        }

        virtual void visit(IWriteVisitor* visitor_) const override
        {
            write(*visitor_, *m_ptr);
        }
        virtual size_t getSize() const override { return m_ptr->size();}
        virtual void setSize(const size_t num) override { num_to_read = num; }
      private:
        const std::map<std::string, V>* m_ptr;
        size_t num_to_read = 0;
    };
}
