#ifndef MO_VISITATION_CONTAINER_TRAITS_HPP
#define MO_VISITATION_CONTAINER_TRAITS_HPP
#include "IDynamicVisitor.hpp"
#include "StructTraits.hpp"

#include <MetaObject/logging/logging.hpp>

#include <ct/type_traits.hpp>

namespace mo
{
    template <class T>
    void resize(T* ptr, size_t size)
    {
        ptr->resize(size);
    }

    template <class T>
    void* values(T* ptr)
    {
        return ptr->data();
    }

    template <class T>
    const void* values(const T* ptr)
    {
        return ptr->data();
    }

    struct MO_EXPORTS ContainerDefault : virtual IContainerTraits
    {
        bool triviallySerializable() const override;

        TypeInfo keyType() const override;
        bool podKeys() const override;

        bool isContinuous() const override;

        size_t getContainerSize(const void* inst) const override;
        void setContainerSize(size_t size, void* inst) const override;

        void* valuePointer(void* inst) const override;
        const void* valuePointer(const void* inst) const override;

        void* keyPointer(void*) const override;
        const void* keyPointer(const void*) const override;
    };

    template <class T, class V, class K = void>
    struct ContainerBase : virtual ContainerDefault
    {
        ContainerBase()
        {
        }

        size_t size() const override
        {
            return sizeof(T);
        }

        bool triviallySerializable() const override
        {
            return std::is_trivially_copyable<T>::value;
        }

        TypeInfo type() const override
        {
            return TypeInfo(typeid(T));
        }

        T* ptr(void* inst) const
        {
            return static_cast<T*>(inst);
        }

        const T* ptr(const void* inst) const
        {
            return static_cast<const T*>(inst);
        }

        T& ref(void* inst) const
        {
            return *ptr(inst);
        }

        const T& ref(const void* inst) const
        {
            return *ptr(inst);
        }

        TypeInfo keyType() const override
        {
            return TypeInfo(typeid(K));
        }

        TypeInfo valueType() const override
        {
            return TypeInfo(typeid(V));
        }

        bool isContinuous() const override
        {
            return true;
        }

        bool podKeys() const override
        {
            return std::is_trivially_copyable<K>::value;
        }

        bool podValues() const override
        {
            return std::is_trivially_copyable<V>::value;
        }
    };
} // namespace mo

#endif // MO_VISITATION_CONTAINER_TRAITS_HPP
