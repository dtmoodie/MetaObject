#ifndef MO_CORE_IOBJECT_TABLE_HPP
#define MO_CORE_IOBJECT_TABLE_HPP
#include <MetaObject/core/export.hpp>

#include <MetaObject/core/detail/ObjectConstructor.hpp>
#include <MetaObject/detail/Export.hpp>
#include <MetaObject/detail/TypeInfo.hpp>
#include <MetaObject/logging/logging.hpp>
#include <RuntimeObjectSystem/IObject.h>
#include <RuntimeObjectSystem/shared_ptr.hpp>

#include <memory>
#include <type_traits>

namespace mo
{
    template <class T>
    using SharedPtrType =
        typename std::conditional<std::is_base_of<IObject, T>::value, rcc::shared_ptr<T>, std::shared_ptr<T>>::type;
    template <class T>
    struct TypeFromSharedPtr;
    template <class T>
    struct TypeFromSharedPtr<std::shared_ptr<T>>
    {
        using type = T;
    };
    template <class T>
    struct TypeFromSharedPtr<rcc::shared_ptr<T>>
    {
        using type = T;
    };

    // interface to object table, not table of IObjects
    struct IObjectTable
    {
        struct MO_EXPORTS IObjectContainer
        {
            using Ptr_t = std::unique_ptr<IObjectContainer>;

            IObjectContainer(const IObjectContainer&) = delete;
            IObjectContainer(IObjectContainer&&) = delete;
            IObjectContainer& operator=(const IObjectContainer&) = delete;
            IObjectContainer& operator=(IObjectContainer&&) = delete;

            IObjectContainer() = default;
            virtual ~IObjectContainer();
        };
        template <class T, class U = T, class... ARGS>
        SharedPtrType<T> getObject(ARGS&&... args);
        template <class T>
        SharedPtrType<T> getObjectOptional();

        template <class PTR>
        void setObject(PTR&& obj);

      protected:
        virtual IObjectContainer* getObjectContainer(mo::TypeInfo) const = 0;
        virtual void setObjectContainer(mo::TypeInfo, std::unique_ptr<IObjectContainer>&&) = 0;

      private:
        template <class T>
        struct TObjectContainer : public IObjectContainer
        {
            using Ptr_t = SharedPtrType<T>;
            TObjectContainer(Ptr_t&& ptr)
                : m_ptr(std::move(ptr))
            {
            }
            TObjectContainer(const Ptr_t& ptr)
                : m_ptr(ptr)
            {
            }
            Ptr_t ptr() const
            {
                return m_ptr;
            }

          private:
            Ptr_t m_ptr;
        };
    };

    /////////////////////////////////////////////////////////////
    /// IMPLEMENTATION
    /////////////////////////////////////////////////////////////

    template <class T, class U, class... ARGS>
    SharedPtrType<T> IObjectTable::getObject(ARGS&&... args)
    {
        SharedPtrType<T> output;
        mo::TypeInfo tinfo = mo::TypeInfo::create<T>();
        auto container = getObjectContainer(tinfo);
        if (!container)
        {
            ObjectConstructor<U> constructor;
            output = constructor.createShared(std::forward<ARGS>(args)...);
            std::unique_ptr<IObjectContainer> owning(new TObjectContainer<T>(output));
            container = owning.get();
            setObjectContainer(tinfo, std::move(owning));
        }
        else
        {
            auto tcontainer = dynamic_cast<TObjectContainer<T>*>(container);
            if (tcontainer)
            {
                output = tcontainer->ptr();
            }
            else
            {
                // This should never happen...
                THROW(warn, "Requested datatype does not match container datatype");
            }
        }
        return output;
    }

    template <class T>
    SharedPtrType<T> IObjectTable::getObjectOptional()
    {
        SharedPtrType<T> output;
        mo::TypeInfo tinfo = mo::TypeInfo::create<T>();
        auto container = getObjectContainer(tinfo);
        if (container)
        {
            auto tcontainer = dynamic_cast<TObjectContainer<T>*>(container);
            if (tcontainer)
            {
                output = tcontainer->ptr();
            }
            else
            {
                // This should never happen...
                THROW(warn, "Requested datatype does not match container datatype");
            }
        }

        return output;
    }

    template <class PTR>
    void IObjectTable::setObject(PTR&& obj)
    {
        using T = typename TypeFromSharedPtr<PTR>::type;
        std::unique_ptr<IObjectContainer> container(new TObjectContainer<T>(std::move(obj)));
        setObjectContainer(mo::TypeInfo::create<T>(), std::move(container));
    }
} // namespace mo

#endif // MO_CORE_IOBJECT_TABLE_HPP
