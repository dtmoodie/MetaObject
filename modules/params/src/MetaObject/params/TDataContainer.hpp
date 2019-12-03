#pragma once
#include "Header.hpp"
#include "IDataContainer.hpp"
#include <MetaObject/core/detail/Allocator.hpp>
#include <MetaObject/runtime_reflection.hpp>
#include <MetaObject/runtime_reflection/type_traits.hpp>
#include <MetaObject/runtime_reflection/visitor_traits/time.hpp>

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

    struct MO_EXPORTS ParamAllocator
    {
        using Ptr_t = std::shared_ptr<ParamAllocator>;
        using ConstPtr_t = std::shared_ptr<const ParamAllocator>;
        struct MO_EXPORTS SerializationBuffer : public ct::TArrayView<uint8_t>
        {
            SerializationBuffer(ParamAllocator& alloc, uint8_t* begin, size_t sz);

            SerializationBuffer(ParamAllocator& alloc, uint8_t* begin, uint8_t* end);

            SerializationBuffer(const SerializationBuffer&) = delete;
            SerializationBuffer(SerializationBuffer&&) noexcept = default;
            SerializationBuffer& operator=(const SerializationBuffer&) = delete;
            SerializationBuffer& operator=(SerializationBuffer&&) noexcept = default;

            ~SerializationBuffer();

          private:
            ParamAllocator& m_alloc;
        };

        static Ptr_t create(Allocator::Ptr_t allocator = Allocator::getDefault());

        ParamAllocator(Allocator::Ptr_t allocator = Allocator::getDefault());

        void setPadding(size_t header, size_t footer = 0);

        template <class T>
        T* allocate(size_t num)
        {
            auto allocation = allocateImpl(num, sizeof(T));
            return ptrCast<T>(allocation.requested);
        }

        template <class T>
        SerializationBuffer allocateSerialization(size_t header_sz, size_t footer_sz, const T* ptr)
        {
            return allocateSerializationImpl(header_sz, footer_sz, static_cast<const void*>(ptr), sizeof(T));
        }

        template <class T>
        void deallocate(T* ptr)
        {
            deallocateImpl(static_cast<void*>(ptr));
        }

        Allocator::Ptr_t getAllocator() const;

        void setAllocator(Allocator::Ptr_t allocator);

      private:
        void deallocateImpl(void* ptr);

        SerializationBuffer
        allocateSerializationImpl(size_t header_sz, size_t footer_sz, const void* ptr, size_t elem_size);

        struct CurrentAllocations
        {
            uint8_t* begin = nullptr;
            uint8_t* requested = nullptr;
            uint8_t* end = nullptr;
            size_t requested_size = 0;
            int ref_count = 1;
        };

        CurrentAllocations allocateImpl(size_t num, size_t elem_size);

        // This is the allocator used for actual allocations
        // IE pinned memory or shared
        Allocator::Ptr_t m_allocator;
        size_t m_header_pad = 0;
        size_t m_footer_pad = 0;
        std::list<CurrentAllocations> m_allocations;
    };

    template <class T>
    struct TDataContainerBase : public IDataContainer
    {
        using type = T;

        TDataContainerBase(const T& data_);
        TDataContainerBase(T&& data_ = T());

        TDataContainerBase(const TDataContainerBase&) = delete;
        TDataContainerBase(TDataContainerBase&&) = delete;
        TDataContainerBase& operator=(const TDataContainerBase&) = delete;
        TDataContainerBase& operator=(TDataContainerBase&&) = delete;

        ~TDataContainerBase() override = default;

        TypeInfo getType() const override;

        void load(ILoadVisitor&) override;
        void save(ISaveVisitor&) const override;
        void load(BinaryInputVisitor& ar) override;
        void save(BinaryOutputVisitor& ar) const override;
        static void visitStatic(StaticVisitor&);
        void visit(StaticVisitor&) const override;

        const Header& getHeader() const override;

        operator std::shared_ptr<T>();
        operator std::shared_ptr<const T>() const;

        operator T*();
        operator const T*() const;

        T* ptr();
        const T* ptr() const;
        std::shared_ptr<T> sharedPtr();
        std::shared_ptr<const T> sharedPtr() const;

        T data;
        Header header;
        std::shared_ptr<void> owning;
    };

    template <class T, class ENABLE = void>
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

    ////////////////////////////////////////////////////////////////////////
    ///  Implementation
    ////////////////////////////////////////////////////////////////////////
    template <class T>
    TDataContainerBase<T>::TDataContainerBase(const T& data_)
        : data(data_)
    {
    }

    template <class T>
    TDataContainerBase<T>::TDataContainerBase(T&& data_)
        : data(std::move(data_))
    {
    }

    template <class T>
    TypeInfo TDataContainerBase<T>::getType() const
    {
        return TypeInfo::create<T>();
    }

    template <class T>
    void TDataContainerBase<T>::load(ILoadVisitor& visitor)
    {
        visitor(&header, "header");
        visitor(&data, "data");
    }

    template <class T>
    void TDataContainerBase<T>::save(ISaveVisitor& visitor) const
    {
        visitor(&header, "header");
        visitor(&data, "data");
    }

    template <class T>
    void TDataContainerBase<T>::visitStatic(StaticVisitor& visitor)
    {
        visitor.template visit<Header>("header");
        visitor.template visit<T>("data");
    }

    template <class T>
    void TDataContainerBase<T>::visit(StaticVisitor& visitor) const
    {
        visitStatic(visitor);
    }

    template <class T>
    void TDataContainerBase<T>::load(BinaryInputVisitor& ar)
    {
        ar(CEREAL_NVP(header));
        ar(CEREAL_NVP(data));
    }

    template <class T>
    void TDataContainerBase<T>::save(BinaryOutputVisitor& ar) const
    {
        ar(CEREAL_NVP(header));
        ar(CEREAL_NVP(data));
    }

    template <class T>
    const Header& TDataContainerBase<T>::getHeader() const
    {
        return header;
    }

    template <class T>
    TDataContainerBase<T>::operator std::shared_ptr<T>()
    {
        auto owning_ptr = shared_from_this();
        return std::shared_ptr<T>(&data, [owning_ptr](T*) {});
    }

    template <class T>
    TDataContainerBase<T>::operator std::shared_ptr<const T>() const
    {
        auto owning_ptr = shared_from_this();
        return std::shared_ptr<const T>(&data, [owning_ptr](T*) {});
    }

    template <class T>
    TDataContainerBase<T>::operator T*()
    {
        return &data;
    }

    template <class T>
    TDataContainerBase<T>::operator const T*() const
    {
        return &data;
    }

    template <class T>
    T* TDataContainerBase<T>::ptr()
    {
        return &data;
    }

    template <class T>
    const T* TDataContainerBase<T>::ptr() const
    {
        return &data;
    }

    template <class T>
    std::shared_ptr<T> TDataContainerBase<T>::sharedPtr()
    {
        auto owning_ptr = shared_from_this();
        return std::shared_ptr<T>(&data, [owning_ptr](T*) {});
    }

    template <class T>
    std::shared_ptr<const T> TDataContainerBase<T>::sharedPtr() const
    {
        auto owning_ptr = shared_from_this();
        return std::shared_ptr<const T>(&data, [owning_ptr](T*) {});
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
