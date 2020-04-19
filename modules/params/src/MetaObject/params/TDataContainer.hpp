#ifndef MO_PARAMS_TDATACONTAINER_HPP
#define MO_PARAMS_TDATACONTAINER_HPP
#include "ContainerTraits.hpp"
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
        using Super_t::ptr;
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

    template <class T>
    struct TDataContainer : public TDataContainerBase<typename ContainerTraits<T>::type, ContainerTraits<T>::CONST>
    {
        using type = typename ContainerTraits<T>::type;
        using Ptr_t = std::shared_ptr<TDataContainer<type>>;
        using ConstPtr_t = std::shared_ptr<const TDataContainer<type>>;
        using Super_t = TDataContainerBase<typename ContainerTraits<T>::type, ContainerTraits<T>::CONST>;

        template <class... ARGS>
        TDataContainer(typename ParamAllocator::Ptr_t alloc, ARGS&&... args);

        TDataContainer();

        ParamAllocator::Ptr_t getAllocator() const;

        void record(IAsyncStream& src) const override;
        void sync(IAsyncStream& dest) const override;

      private:
        ContainerTraits<T> m_traits;
    };

    template <class T>
    using vector = std::vector<T, TStlAllocator<T>>;

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

    template <class T>
    template <class... ARGS>
    TDataContainer<T>::TDataContainer(typename ParamAllocator::Ptr_t alloc, ARGS&&... args)
        : m_traits(alloc)
        , Super_t(ContainerTraits<T>::staticCreate(alloc, std::forward<ARGS>(args)...))
    {
    }

    template <class T>
    TDataContainer<T>::TDataContainer()
    {
    }

    template <class T>
    ParamAllocator::Ptr_t TDataContainer<T>::getAllocator() const
    {
        return m_traits.getAllocator();
    }

template <class T>
    void TDataContainer<T>::record(IAsyncStream& src) const
    {
        m_traits.record(src);
    }

    template <class T>
    void TDataContainer<T>::sync(IAsyncStream& dest) const
    {
        m_traits.sync(dest);
    }

    template <class T>
    T* IDataContainer::ptr()
    {
        if (getType().isType<T>())
        {
            auto typed = static_cast<TDataContainer<T>*>(this);
            if (typed)
            {
                return &typed->data;
            }
        }
        return nullptr;
    }

    template <class T>
    const T* IDataContainer::ptr() const
    {
        if (getType().isType<T>())
        {
            auto typed = static_cast<const TDataContainer<T>*>(this);
            if (typed)
            {
                return &typed->data;
            }
        }
        return nullptr;
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
#endif // MO_PARAMS_TDATACONTAINER_HPP