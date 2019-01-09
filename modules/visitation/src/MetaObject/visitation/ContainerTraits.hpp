#ifndef MO_VISITATION_CONTAINER_TRAITS_HPP
#define MO_VISITATION_CONTAINER_TRAITS_HPP
#include "IDynamicVisitor.hpp"
#include <MetaObject/logging/logging.hpp>
#include <ct/TypeTraits.hpp>
namespace mo
{
    template <class T>
    struct KeyType
    {
        using type = void;
    };
    template <class T>
    struct ValueType
    {
        using type = typename T::value_type;
    };
    template <class T>
    struct IsContinuous;

    template <class T>
    size_t getSize(const T& container);
    template <class T>
    void resize(T& container, const size_t size);

    template <class T>
    struct ContainerBase : virtual public ILoadContainerTraits
    {
        using base = ILoadContainerTraits;

        ContainerBase(T* ptr);

        // ITraits
        void load(ILoadVisitor* visitor) override;
        void save(ISaveVisitor* visitor) const override;
        void visit(StaticVisitor* visitor) const override;
        TypeInfo type() const override;

        // IContainerTraits
        TypeInfo keyType() const override;
        TypeInfo valueType() const override;
        bool isContinuous() const override;
        bool podValues() const override;
        bool podKeys() const override;
        size_t getSize() const override;
        void setSize(const size_t num) override;
        void setInstance(void* ptr, const TypeInfo type_);
        void setInstance(const void* ptr, const TypeInfo type_);

      private:
        T* m_ptr;
    };

    // const specialization
    template <class T>
    struct ContainerBase<const T> : virtual public ISaveContainerTraits
    {
        using base = ISaveContainerTraits;

        ContainerBase(const T* ptr);

        // ITraits
        void save(ISaveVisitor* visitor) const override;
        void visit(StaticVisitor* visitor) const override;
        TypeInfo type() const override;

        // IContainerTraits
        TypeInfo keyType() const override;
        TypeInfo valueType() const override;
        bool isContinuous() const override;
        bool podValues() const override;
        bool podKeys() const override;
        size_t getSize() const override;
        void setInstance(const void* ptr, const TypeInfo type_);

      private:
        const T* m_ptr;
    };

    //////////////////////////////////////////////////////////////////////////////////////////////////
    ///             IMPLEMENTATION
    //////////////////////////////////////////////////////////////////////////////////////////////////
    template <class T>
    size_t getSize(const T& container)
    {
        return container.size();
    }
    template <class T>
    void resize(T& container, const size_t size)
    {
        container.resize(size);
    }

    template <class T>
    ContainerBase<T>::ContainerBase(T* ptr)
        : m_ptr(ptr)
    {
    }

    // ITraits
    template <class T>
    void ContainerBase<T>::load(ILoadVisitor* visitor)
    {
        Visit<T>::load(*visitor, m_ptr, "", 1);
    }

    template <class T>
    void ContainerBase<T>::save(ISaveVisitor* visitor) const
    {
        Visit<T>::save(*visitor, m_ptr, "", 1);
    }

    template <class T>
    void ContainerBase<T>::visit(StaticVisitor* visitor) const
    {
        if (!std::is_same<typename KeyType<T>::type, void>::value)
        {
            visitor->template visit<typename KeyType<T>::type>("key", 1);
        }
        visitor->template visit<T>("value", 1);
    }

    template <class T>
    TypeInfo ContainerBase<T>::keyType() const
    {
        return TypeInfo(typeid(typename KeyType<T>::type));
    }

    template <class T>
    TypeInfo ContainerBase<T>::valueType() const
    {
        return TypeInfo(typeid(typename ValueType<T>::type));
    }

    template <class T>
    bool ContainerBase<T>::isContinuous() const
    {
        return IsContinuous<T>::value;
    }

    template <class T>
    bool ContainerBase<T>::podValues() const
    {
        return std::is_pod<typename ValueType<T>::type>::value;
    }

    template <class T>
    bool ContainerBase<T>::podKeys() const
    {
        return std::is_pod<typename KeyType<T>::type>::value;
    }

    template <class T>
    size_t ContainerBase<T>::getSize() const
    {
        return mo::getSize(*m_ptr);
    }

    template <class T>
    void ContainerBase<T>::setSize(const size_t num)
    {
        if (m_ptr)
        {
            resize(*m_ptr, num);
        }
    }

    template <class T>
    TypeInfo ContainerBase<T>::type() const
    {
        return TypeInfo(typeid(T));
    }

    template <class T>
    void ContainerBase<T>::setInstance(void* ptr, const TypeInfo type_)
    {
        MO_ASSERT(type_ == type());
        m_ptr = static_cast<T*>(ptr);
    }

    template <class T>
    void ContainerBase<T>::setInstance(const void*, const TypeInfo)
    {
        THROW(warn, "Can't accept a const void*");
    }

    // const specialization
    template <class T>
    ContainerBase<const T>::ContainerBase(const T* ptr)
        : m_ptr(ptr)
    {
    }

    template <class T>
    void ContainerBase<const T>::save(ISaveVisitor* visitor) const
    {
        Visit<T>::save(*visitor, m_ptr, "", 1);
    }

    template <class T>
    void ContainerBase<const T>::visit(StaticVisitor* visitor) const
    {
        if (!std::is_same<typename KeyType<T>::type, void>::value)
        {
            visitor->template visit<typename KeyType<T>::type>("key", 1);
        }
        visitor->template visit<typename ValueType<T>::type>("value", 1);
    }

    template <class T>
    TypeInfo ContainerBase<const T>::keyType() const
    {
        return TypeInfo(typeid(typename KeyType<T>::type));
    }

    template <class T>
    TypeInfo ContainerBase<const T>::valueType() const
    {
        return TypeInfo(typeid(typename ValueType<T>::type));
    }

    template <class T>
    bool ContainerBase<const T>::isContinuous() const
    {
        return IsContinuous<T>::value;
    }

    template <class T>
    bool ContainerBase<const T>::podValues() const
    {
        return std::is_pod<typename ValueType<T>::type>::value;
    }

    template <class T>
    bool ContainerBase<const T>::podKeys() const
    {
        return std::is_pod<typename KeyType<T>::type>::value;
    }

    template <class T>
    size_t ContainerBase<const T>::getSize() const
    {
        if (m_ptr)
        {
            return mo::getSize(*m_ptr);
        }
        return 0;
    }

    template <class T>
    void ContainerBase<const T>::setInstance(const void*, const TypeInfo)
    {
        THROW(warn, "Can't accept a const void*");
    }

    template <class T>
    TypeInfo ContainerBase<const T>::type() const
    {
        return TypeInfo(typeid(T));
    }
}

#endif // MO_VISITATION_CONTAINER_TRAITS_HPP
