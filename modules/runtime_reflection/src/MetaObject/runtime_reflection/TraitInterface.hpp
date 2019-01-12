#ifndef MO_VISITATION_TRAIT_INTERFACE_HPP
#define MO_VISITATION_TRAIT_INTERFACE_HPP
#include <MetaObject/detail/TypeInfo.hpp>

namespace mo
{
    struct IDynamicVisitor;
    struct ILoadVisitor;
    struct ISaveVisitor;
    // visit an object without creating an object
    struct StaticVisitor;

    struct ITraits
    {
        virtual ~ITraits();
        virtual TypeInfo type() const = 0;
        virtual std::string getName() const;
        virtual void visit(StaticVisitor* visitor) const = 0;
    };

    struct ISaveTraits
    {
        virtual ~ISaveTraits();
        virtual void save(ISaveVisitor* visitor) const = 0;

      protected:
        virtual void setInstance(const void* ptr, const TypeInfo type) = 0;
    };

    struct ILoadTraits
    {
        virtual ~ILoadTraits();
        virtual void load(ILoadVisitor* visitor) = 0;

      protected:
        virtual void setInstance(void* ptr, const TypeInfo type) = 0;
    };

    struct IStructTraits : virtual public ITraits
    {
        // sizeof(T)
        virtual size_t size() const = 0;
        // can be serialized via a memcpy(ptr)
        virtual bool triviallySerializable() const = 0;
        // if it can be serialized by one of the primitive supported types, such as
        // struct{float x,y,z;} can be serialized as 3 floats in continuous memory
        virtual bool isPrimitiveType() const = 0;
    };

    struct ISaveStructTraits : virtual public IStructTraits, virtual public ISaveTraits
    {
        // const ptr to type
        virtual const void* ptr() const = 0;
        virtual size_t count() const = 0;
        virtual void increment() = 0;

        template <class T>
        void setInstance(const T* ptr)
        {
            ISaveTraits::setInstance(static_cast<const void*>(ptr), TypeInfo(typeid(T)));
        }
    };

    struct ILoadStructTraits : virtual public ISaveStructTraits, virtual public ILoadTraits
    {
        virtual const void* ptr() const = 0;
        // non const ptr to type
        virtual void* ptr() = 0;
        template <class T>
        void setInstance(T* ptr)
        {
            ILoadTraits::setInstance(static_cast<void*>(ptr), TypeInfo(typeid(T)));
        }
    };

    struct IContainerTraits : virtual public ITraits
    {
        virtual TypeInfo keyType() const = 0;
        virtual TypeInfo valueType() const = 0;

        virtual bool isContinuous() const = 0;
        virtual bool podValues() const = 0;
        virtual bool podKeys() const = 0;
    };

    struct ISaveContainerTraits : virtual public IContainerTraits, virtual public ISaveTraits
    {
        virtual size_t getSize() const = 0;
    };

    struct ILoadContainerTraits : virtual public ISaveContainerTraits, virtual public ILoadTraits
    {
        virtual void setSize(const size_t num) = 0;
    };

    template <class T>
    struct ArrayContainerTrait;

    template <class T, class E = void>
    struct TTraits;
}

#endif // MO_VISITATION_TRAIT_INTERFACE_HPP
