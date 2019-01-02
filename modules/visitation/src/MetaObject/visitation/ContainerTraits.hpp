#ifndef MO_VISITATION_CONTAINER_TRAITS_HPP
#define MO_VISITATION_CONTAINER_TRAITS_HPP
#include "IDynamicVisitor.hpp"
#include <ct/TypeTraits.hpp>

namespace mo
{
    template<class T> struct KeyType{using type = void;};
    template<class T> struct ValueType{using type = typename T::value_type;};
    template<class T> struct IsContinuous;

    template<class T> size_t getSize(const T& container);
    template<class T> void resize(T& container, const size_t size);

    template<class T>
    struct ContainerBase: public IContainerTraits
    {
        using base = IContainerTraits;

        ContainerBase(T* ptr);

        // ITraits
         void visit(IReadVisitor* visitor) override;
         void visit(IWriteVisitor* visitor) const override;
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

    private:
        T* m_ptr;
    };

    // const specialization
    template<class T>
    struct ContainerBase<const T>: public IContainerTraits
    {
        using base = IContainerTraits;

        ContainerBase(const T* ptr);

        // ITraits
        void visit(IReadVisitor* visitor) override;
        void visit(IWriteVisitor* visitor) const override;
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

    private:
        const T* m_ptr;
    };



    //////////////////////////////////////////////////////////////////////////////////////////////////
    ///             IMPLEMENTATION
    //////////////////////////////////////////////////////////////////////////////////////////////////
    template<class T> size_t getSize(const T& container){return container.size();}
    template<class T> void resize(T& container, const size_t size){container.resize(size);}

    template<class T>
    ContainerBase<T>::ContainerBase(T* ptr):
        m_ptr(ptr)
    {

    }

    // ITraits
    template<class T>
    void ContainerBase<T>::visit(IReadVisitor* visitor)
    {
        Visit<T>::read(*visitor, m_ptr, "", 1);
    }

    template<class T>
    void ContainerBase<T>::visit(IWriteVisitor* visitor) const
    {
        Visit<T>::write(*visitor, m_ptr, "", 1);
    }

    template<class T>
    void ContainerBase<T>::visit(StaticVisitor* visitor) const
    {
        if(!std::is_same<typename KeyType<T>::type, void>::value)
        {
            visitor->template visit<typename KeyType<T>::type>("key", 1);
        }
        visitor->template visit<T>("value", 1);
    }

    template<class T>
    TypeInfo ContainerBase<T>::keyType() const
    {
        return TypeInfo(typeid(typename KeyType<T>::type));
    }

    template<class T>
    TypeInfo ContainerBase<T>::valueType() const
    {
        return TypeInfo(typeid(typename ValueType<T>::type));
    }

    template<class T>
    bool ContainerBase<T>::isContinuous() const
    {
        return IsContinuous<T>::value;
    }

    template<class T>
    bool ContainerBase<T>::podValues() const
    {
        return std::is_pod<typename ValueType<T>::type>::value;
    }

    template<class T>
    bool ContainerBase<T>::podKeys() const
    {
        return std::is_pod<typename KeyType<T>::type>::value;
    }

    template<class T>
    size_t ContainerBase<T>::getSize() const
    {
        if(m_ptr)
        {
            return mo::getSize(*m_ptr);
        }else
        {
            return 0;
        }
    }

    template<class T>
    void ContainerBase<T>::setSize(const size_t num)
    {
        if(m_ptr)
        {
            resize(*m_ptr, num);
        }
    }

    template<class T>
    TypeInfo ContainerBase<T>::type() const
    {
        return TypeInfo(typeid(T));
    }

    // const specialization
    template<class T>
    ContainerBase<const T>::ContainerBase(const T* ptr):
        m_ptr(ptr)
    {

    }

    template<class T>
    void ContainerBase<const T>::visit(IReadVisitor*)
    {
        throw std::runtime_error("Attempted to read data into a const container");
    }

    template<class T>
    void ContainerBase<const T>::visit(IWriteVisitor* visitor) const
    {
        Visit<T>::write(*visitor, m_ptr, "", 1);
    }

    template<class T>
    void ContainerBase<const T>::visit(StaticVisitor* visitor) const
    {
        if(!std::is_same<typename KeyType<T>::type, void>::value)
        {
            visitor->template visit<typename KeyType<T>::type>("key", 1);
        }
        visitor->template visit<typename ValueType<T>::type>("value", 1);
    }

    template<class T>
    TypeInfo ContainerBase<const T>::keyType() const
    {
        return TypeInfo(typeid(typename KeyType<T>::type));
    }

    template<class T>
    TypeInfo ContainerBase<const T>::valueType() const
    {
        return TypeInfo(typeid(typename ValueType<T>::type));
    }

    template<class T>
    bool ContainerBase<const T>::isContinuous() const
    {
        return IsContinuous<T>::value;
    }

    template<class T>
    bool ContainerBase<const T>::podValues() const
    {
        return std::is_pod<typename ValueType<T>::type>::value;
    }

    template<class T>
    bool ContainerBase<const T>::podKeys() const
    {
        return std::is_pod<typename KeyType<T>::type>::value;
    }

    template<class T>
    size_t ContainerBase<const T>::getSize() const
    {
        if(m_ptr)
        {
            return mo::getSize(*m_ptr);
        }else
        {
            return 0;
        }
    }

    template<class T>
    void ContainerBase<const T>::setSize(const size_t)
    {
        throw std::runtime_error("Attempted to resize a const container");
    }

    template<class T>
    TypeInfo ContainerBase<const T>::type() const
    {
        return TypeInfo(typeid(T));
    }
}

#endif // MO_VISITATION_CONTAINER_TRAITS_HPP
