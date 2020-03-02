#pragma once
#include "Header.hpp"
#include "IDataContainer.hpp"
#include "ParamAllocator.hpp"

#include <MetaObject/core/detail/Allocator.hpp>
#include <MetaObject/runtime_reflection.hpp>
#include <MetaObject/runtime_reflection/type_traits.hpp>
#include <MetaObject/runtime_reflection/visitor_traits/time.hpp>
#include <MetaObject/thread/fiber_include.hpp>

#include <ct/reflect/cerealize.hpp>

#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>

#include <boost/optional.hpp>
#include <list>
namespace mo
{
    using ct::ptrCast;

    // This is the allocator that is owned by TParam, it is passed into each container as they are created
    // so that allocation properties can be consistent for each parameter.
    // IE a parameter that gets serialized

    template <class T>
    struct TDataContainerConstBase : public IDataContainer
    {
        using type = T;

        TDataContainerConstBase(const T& data_)
            : data(data_)
        {
        }
        TDataContainerConstBase(T&& data_ = T())
            : data(std::move(data_))
        {
        }

        TDataContainerConstBase(const TDataContainerConstBase&) = delete;
        TDataContainerConstBase(TDataContainerConstBase&&) = delete;
        TDataContainerConstBase& operator=(const TDataContainerConstBase&) = delete;
        TDataContainerConstBase& operator=(TDataContainerConstBase&&) = delete;
        ~TDataContainerConstBase() override = default;

        TypeInfo getType() const override;
        void save(ISaveVisitor&) const override;
        void save(BinaryOutputVisitor& ar) const override;
        static void visitStatic(StaticVisitor&);
        void visit(StaticVisitor&) const override;

        const Header& getHeader() const override;

        operator std::shared_ptr<const T>() const;
        operator const T*() const;
        const T* ptr() const;
        std::shared_ptr<const T> sharedPtr() const;

        T data;
        Header header;
        std::shared_ptr<void> owning;
    };

    template <class T>
    struct TDataContainerNonConstBase : public TDataContainerConstBase<T>
    {
        using Super_t = TDataContainerConstBase<T>;
        using type = typename Super_t::type;

        TDataContainerNonConstBase(const T& data_)
            : Super_t(data_)
        {
        }
        TDataContainerNonConstBase(T&& data_ = T())
            : Super_t(std::move(data_))
        {
        }

        ~TDataContainerNonConstBase() override = default;

        void load(ILoadVisitor&) override;
        void load(BinaryInputVisitor& ar) override;
        operator std::shared_ptr<T>();
        operator T*();
        T* ptr();
        std::shared_ptr<T> sharedPtr();
    };

    template <class T, bool Const = false>
    struct TDataContainerBase : TDataContainerNonConstBase<T>
    {
        using Super_t = TDataContainerNonConstBase<T>;
        using type = typename Super_t::type;

        TDataContainerBase(const T& data_)
            : Super_t(data_)
        {
        }

        TDataContainerBase(T&& data_ = T())
            : Super_t(std::move(data_))
        {
        }
    };

    template <class T>
    struct TDataContainerBase<T, true> : TDataContainerConstBase<T>
    {
        using Super_t = TDataContainerConstBase<T>;
        using type = typename Super_t::type;

        TDataContainerBase(const T& data_)
            : Super_t(data_)
        {
        }

        TDataContainerBase(T&& data_ = T())
            : Super_t(std::move(data_))
        {
        }

        void load(ILoadVisitor&) override
        {
        }
        void load(BinaryInputVisitor& ar) override
        {
        }
    };

    template <class T, class ENABLE>
    struct TDataContainer : public TDataContainerBase<T>
    {
        using type = T;
        using Ptr_t = std::shared_ptr<TDataContainer<type>>;
        using ConstPtr_t = std::shared_ptr<const TDataContainer<type>>;
        using Super_t = TDataContainerBase<T>;

        template <class... ARGS>
        TDataContainer(typename ParamAllocator::Ptr_t, ARGS&&... args)
            : Super_t(T(std::forward<ARGS>(args)...))
        {
        }
    };

    template <class T>
    class TVectorAllocator
    {
      public:
        using value_type = T;
        using pointer = T*;
        using const_pointer = const T*;
        using reference = T&;
        using const_reference = const T&;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        template <class U>
        struct rebind
        {
            using other = TVectorAllocator<U>;
        };

        TVectorAllocator(typename ParamAllocator::Ptr_t allocator = ParamAllocator::create())
            : m_allocator(std::move(allocator))
        {
        }

        TVectorAllocator(TVectorAllocator<T>&& other) = default;
        TVectorAllocator(const TVectorAllocator<T>& other) = default;

        template <class U>
        TVectorAllocator(TVectorAllocator<U>&& other, ct::EnableIf<sizeof(U) == sizeof(T), const void*> = nullptr)
        {
            m_allocator = other.getAllocator();
        }

        template <class U>
        TVectorAllocator(TVectorAllocator<U>&& other, ct::EnableIf<sizeof(U) != sizeof(T), const void*> = nullptr)
        {
            auto alloc = other.getAllocator();
            if (alloc)
            {
                m_allocator = ParamAllocator::create(alloc->getAllocator());
            }
        }

        template <class U>
        TVectorAllocator(const TVectorAllocator<U>& other, ct::EnableIf<sizeof(U) == sizeof(T), const void*> = nullptr)
            : m_allocator(other.getAllocator())
        {
        }

        template <class U>
        TVectorAllocator(const TVectorAllocator<U>& other, ct::EnableIf<sizeof(U) != sizeof(T), const void*> = nullptr)
        {
            auto alloc = other.getAllocator();
            if (alloc)
            {
                m_allocator = ParamAllocator::create(alloc->getAllocator());
            }
        }

        pointer allocate(size_type n, std::allocator<void>::const_pointer)
        {
            return allocate(n);
        }

        pointer allocate(size_type n)
        {
            auto output = m_allocator->allocate<T>(n);
            return output;
        }

        void deallocate(pointer ptr, size_type)
        {
            if (m_allocator)
            {
                m_allocator->deallocate(ptr);
            }
        }

        typename ParamAllocator::Ptr_t getAllocator() const
        {
            return m_allocator;
        }

        bool operator==(const TVectorAllocator<T>& rhs) const
        {
            return this->m_allocator == rhs.m_allocator;
        }

        bool operator!=(const TVectorAllocator<T>& rhs) const
        {
            return this->m_allocator != rhs.m_allocator;
        }

      private:
        typename ParamAllocator::Ptr_t m_allocator;
    };

    template <class T>
    using vector = std::vector<T, TVectorAllocator<T>>;

    template <class T, class A>
    struct TDataContainer<std::vector<T, A>, ct::EnableIf<std::is_trivially_copyable<T>::value>>
        : public TDataContainerBase<std::vector<T, TVectorAllocator<T>>>
    {
        using Super_t = TDataContainerBase<std::vector<T, TVectorAllocator<T>>>;
        using type = std::vector<T, TVectorAllocator<T>>;
        using Ptr_t = std::shared_ptr<TDataContainer<std::vector<T, A>>>;
        using ConstPtr_t = std::shared_ptr<const TDataContainer<std::vector<T, A>>>;

        TDataContainer()
            : m_allocator(ParamAllocator::create())
        {
        }

        template <class... ARGS>
        TDataContainer(ParamAllocator::Ptr_t allocator, ARGS&&... args)
            : m_allocator(allocator)
            , Super_t(type(std::forward<ARGS>(args)..., TVectorAllocator<T>(allocator)))
        {
        }

        ParamAllocator::Ptr_t getAllocator() const
        {
            return m_allocator;
        }

      private:
        ParamAllocator::Ptr_t m_allocator;
    };

    template <class T>
    struct TDataContainer<ct::TArrayView<const T>, void> : public TDataContainerBase<ct::TArrayView<const T>, true>
    {
        using Super_t = TDataContainerBase<ct::TArrayView<const T>, true>;
        using type = typename Super_t::type;

        template <class... ARGS>
        TDataContainer(typename ParamAllocator::Ptr_t, ARGS&&... args)
            : Super_t(ct::TArrayView<const T>(std::forward<ARGS>(args)...))
        {
        }
    };

    ////////////////////////////////////////////////////////////////////////
    ///  Implementation
    ////////////////////////////////////////////////////////////////////////

    template <class T>
    TypeInfo TDataContainerConstBase<T>::getType() const
    {
        return TypeInfo::create<T>();
    }

    template <class T>
    void TDataContainerConstBase<T>::save(ISaveVisitor& visitor) const
    {
        visitor(&header, "header");
        visitor(&data, "data");
    }

    template <class T>
    void TDataContainerConstBase<T>::save(BinaryOutputVisitor& ar) const
    {
        ar(CEREAL_NVP(header));
        ar(CEREAL_NVP(data));
    }

    template <class T>
    void TDataContainerConstBase<T>::visitStatic(StaticVisitor& visitor)
    {
        visitor.template visit<Header>("header");
        visitor.template visit<T>("data");
    }

    template <class T>
    void TDataContainerConstBase<T>::visit(StaticVisitor& visitor) const
    {
        visitStatic(visitor);
    }

    template <class T>
    void TDataContainerNonConstBase<T>::load(BinaryInputVisitor& ar)
    {
        ar(CEREAL_NVP(this->header));
        ar(CEREAL_NVP(this->data));
    }

    template <class T>
    const Header& TDataContainerConstBase<T>::getHeader() const
    {
        return header;
    }

    template <class T>
    TDataContainerConstBase<T>::operator std::shared_ptr<const T>() const
    {
        auto owning_ptr = shared_from_this();
        return std::shared_ptr<const T>(&data, [owning_ptr](T*) {});
    }

    template <class T>
    TDataContainerConstBase<T>::operator const T*() const
    {
        return &data;
    }

    template <class T>
    const T* TDataContainerConstBase<T>::ptr() const
    {
        return &data;
    }

    template <class T>
    std::shared_ptr<const T> TDataContainerConstBase<T>::sharedPtr() const
    {
        auto owning_ptr = shared_from_this();
        return std::shared_ptr<const T>(&data, [owning_ptr](T*) {});
    }

    template <class T>
    void TDataContainerNonConstBase<T>::load(ILoadVisitor& visitor)
    {
        visitor(&this->header, "header");
        visitor(&this->data, "data");
    }

    template <class T>
    TDataContainerNonConstBase<T>::operator std::shared_ptr<T>()
    {
        return sharedPtr();
    }

    template <class T>
    TDataContainerNonConstBase<T>::operator T*()
    {
        return &this->data;
    }

    template <class T>
    T* TDataContainerNonConstBase<T>::ptr()
    {
        return &this->data;
    }

    template <class T>
    std::shared_ptr<T> TDataContainerNonConstBase<T>::sharedPtr()
    {
        auto owning_ptr = this->shared_from_this();
        return std::shared_ptr<T>(&this->data, [owning_ptr](T*) {});
    }
} // namespace mo

namespace cereal
{
    //! Saving for boost::optional
    template <class Archive, class Optioned>
    inline void save(Archive& ar, ::boost::optional<Optioned> const& optional)
    {
        bool init_flag(optional);
        if (init_flag)
        {
            ar(cereal::make_nvp("initialized", true));
            ar(cereal::make_nvp("value", optional.get()));
        }
        else
        {
            ar(cereal::make_nvp("initialized", false));
        }
    }

    //! Loading for boost::optional
    template <class Archive, class Optioned>
    inline void load(Archive& ar, ::boost::optional<Optioned>& optional)
    {

        bool init_flag;
        ar(cereal::make_nvp("initialized", init_flag));
        if (init_flag)
        {
            Optioned val;
            ar(cereal::make_nvp("value", val));
            optional = val;
        }
        else
        {
            optional = ::boost::none; // this is all we need to do to reset the internal flag and value
        }
    }
} // namespace cereal
