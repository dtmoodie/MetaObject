#ifndef MO_VISITATION_IDYNAMICVISITOR_HPP
#define MO_VISITATION_IDYNAMICVISITOR_HPP

#include "TraitInterface.hpp"
#include "type_traits.hpp"

#include <MetaObject/detail/TypeInfo.hpp>

#include <ct/type_traits.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>

namespace mo
{
    class Allocator;
    struct MO_EXPORTS CacheDataContainer
    {
        CacheDataContainer() = default;
        CacheDataContainer(const CacheDataContainer&) = default;
        CacheDataContainer(CacheDataContainer&&) = default;
        CacheDataContainer& operator=(const CacheDataContainer&) = default;
        CacheDataContainer& operator=(CacheDataContainer&&) = default;

        virtual ~CacheDataContainer();
        TypeInfo type;
    };

    template <class T>
    struct DataContainer : public CacheDataContainer
    {
        DataContainer(T&& val)
            : m_val(std::move(val))
        {
            type = type.template create<T>();
        }
        T m_val;
    };

    template <class T>
    char is_complete_impl(char (*)[sizeof(T)]);

    template <class>
    char (&is_complete_impl(...))[2];

    template <class T>
    struct is_complete
    {
        enum
        {
            value = sizeof(is_complete_impl<T>(0)) == sizeof(char)
        };
    };
    template <class T, class U = void>
    using EnableIfTraitExists = ct::EnableIf<!TTraits<T>::DEFAULT, U>;

    template <class T, class U = void>
    using EnableIfNotTraitExists = ct::EnableIf<TTraits<T>::DEFAULT, U>;

    struct VisitorTraits
    {
        // if true, the name field of the () operator is used to search for the provided data
        bool supports_named_access;
        // If this is true, read data from external source and put into the visited struct
        // else read data from struct and put into output
        bool reader;
    };

    struct MO_EXPORTS IDynamicVisitor
    {
        IDynamicVisitor() = default;
        IDynamicVisitor(const IDynamicVisitor&) = default;
        IDynamicVisitor(IDynamicVisitor&&) = default;
        IDynamicVisitor& operator=(const IDynamicVisitor&) = default;
        IDynamicVisitor& operator=(IDynamicVisitor&&) = default;

        virtual ~IDynamicVisitor();

        virtual VisitorTraits traits() const = 0;

        template <class T>
        void pushCach(T&& val, const std::string& name, uint64_t id = 0);

        template <class T>
        bool tryPopCache(T& val, const std::string& name, uint64_t id = 0);

        template <class T>
        T popCache(const std::string& name, uint64_t id = 0);

        virtual std::shared_ptr<Allocator> getAllocator() const = 0;
        virtual void setAllocator(std::shared_ptr<Allocator>) = 0;

      protected:
        virtual std::unique_ptr<CacheDataContainer>& accessCache(const std::string& name, uint64_t id = 0) = 0;
    };

    struct MO_EXPORTS ILoadVisitor : public virtual IDynamicVisitor
    {
        virtual ILoadVisitor& operator()(char* val, const std::string& name = "", size_t cnt = 1) = 0;
        virtual ILoadVisitor& operator()(bool* val, const std::string& name = "", size_t cnt = 1) = 0;
        virtual ILoadVisitor& operator()(int8_t* val, const std::string& name = "", size_t cnt = 1) = 0;
        virtual ILoadVisitor& operator()(uint8_t* val, const std::string& name = "", size_t cnt = 1) = 0;
        virtual ILoadVisitor& operator()(int16_t* val, const std::string& name = "", size_t cnt = 1) = 0;
        virtual ILoadVisitor& operator()(uint16_t* val, const std::string& name = "", size_t cnt = 1) = 0;
        virtual ILoadVisitor& operator()(int32_t* val, const std::string& name = "", size_t cnt = 1) = 0;
        virtual ILoadVisitor& operator()(uint32_t* val, const std::string& name = "", size_t cnt = 1) = 0;
        virtual ILoadVisitor& operator()(int64_t* val, const std::string& name = "", size_t cnt = 1) = 0;
        virtual ILoadVisitor& operator()(uint64_t* val, const std::string& name = "", size_t cnt = 1) = 0;
#ifdef ENVIRONMENT64
#ifndef _MSC_VER
        virtual ILoadVisitor& operator()(long long* val, const std::string& name = "", size_t cnt = 1) = 0;
        virtual ILoadVisitor& operator()(unsigned long long* val, const std::string& name = "", size_t cnt = 1) = 0;
#endif
#else
        virtual ILoadVisitor& operator()(long int* val, const std::string& name = "", const size_t cnt = 1) = 0;
        virtual ILoadVisitor&
        operator()(unsigned long int* val, const std::string& name = "", const size_t cnt = 1) = 0;
#endif

        virtual ILoadVisitor& operator()(float* val, const std::string& name = "", size_t cnt = 1) = 0;
        virtual ILoadVisitor& operator()(double* val, const std::string& name = "", size_t cnt = 1) = 0;
        virtual ILoadVisitor& operator()(void* binary, const std::string& name = "", size_t num_bytes = 1) = 0;

        template <class T>
        ILoadVisitor& operator()(T* val, const std::string& name = "", size_t cnt = 1);

        template <class T>
        T* getPointer(uint32_t id);

        template <class T>
        void setSerializedPointer(T* ptr, uint32_t id);

        virtual std::string getCurrentElementName() const = 0;
        virtual size_t getCurrentContainerSize() const = 0;

        virtual std::shared_ptr<Allocator> getAllocator() const = 0;
        virtual void setAllocator(std::shared_ptr<Allocator>) = 0;

      protected:
        virtual ILoadVisitor&
        operator()(const IStructTraits* trait, void* inst, const std::string& name = "", size_t cnt = 1) = 0;
        virtual ILoadVisitor&
        operator()(const IContainerTraits* trait, void* inst, const std::string& name = "", size_t cnt = 1) = 0;

        // These are helpers that enforce casting of a trait to the ILoadStructTraits type instead of infinitely going
        // into the templated () operator above
        ILoadVisitor& loadTrait(const IStructTraits* trait, void* inst, const std::string& name = "", size_t cnt = 1);
        ILoadVisitor&
        loadTrait(const IContainerTraits* trait, void* inst, const std::string& name = "", size_t cnt = 1);

        virtual void* getPointer(TypeInfo type, uint32_t id) = 0;
        virtual void setSerializedPointer(TypeInfo type, uint32_t id, void* ptr) = 0;
    };

    using PrimiviteRuntimeTypes = ct::VariadicTypedef<char,
                                                      bool,
                                                      int8_t,
                                                      uint8_t,
                                                      int16_t,
                                                      uint16_t,
                                                      int32_t,
                                                      uint32_t,
                                                      int64_t,
                                                      uint64_t,
                                                      float,
                                                      double>;
    template <class T>
    struct IsPrimitiveRuntimeReflected
    {

        static constexpr const bool value = PrimiviteRuntimeTypes::contains<T>();
    };

    struct MO_EXPORTS ISaveVisitor : public virtual IDynamicVisitor
    {
        virtual ISaveVisitor& operator()(const bool* val, const std::string& name = "", size_t cnt = 1) = 0;
        virtual ISaveVisitor& operator()(const char* val, const std::string& name = "", size_t cnt = 1) = 0;
        virtual ISaveVisitor& operator()(const int8_t* val, const std::string& name = "", size_t cnt = 1) = 0;
        virtual ISaveVisitor& operator()(const uint8_t* val, const std::string& name = "", size_t cnt = 1) = 0;
        virtual ISaveVisitor& operator()(const int16_t* val, const std::string& name = "", size_t cnt = 1) = 0;
        virtual ISaveVisitor& operator()(const uint16_t* val, const std::string& name = "", size_t cnt = 1) = 0;
        virtual ISaveVisitor& operator()(const int32_t* val, const std::string& name = "", size_t cnt = 1) = 0;
        virtual ISaveVisitor& operator()(const uint32_t* val, const std::string& name = "", size_t cnt = 1) = 0;
        virtual ISaveVisitor& operator()(const int64_t* val, const std::string& name = "", size_t cnt = 1) = 0;
        virtual ISaveVisitor& operator()(const uint64_t* val, const std::string& name = "", size_t cnt = 1) = 0;
#ifdef ENVIRONMENT64
#ifndef _MSC_VER
        virtual ISaveVisitor& operator()(const long long* val, const std::string& name = "", size_t cnt = 1) = 0;
        virtual ISaveVisitor&
        operator()(const unsigned long long* val, const std::string& name = "", size_t cnt = 1) = 0;
#endif
#else
        virtual ISaveVisitor& operator()(const long int* val, const std::string& name = "", size_t cnt = 1) = 0;
        virtual ISaveVisitor&
        operator()(const unsigned long int* val, const std::string& name = "", size_t cnt = 1) = 0;
#endif
        virtual ISaveVisitor& operator()(const float* val, const std::string& name = "", size_t cnt = 1) = 0;
        virtual ISaveVisitor& operator()(const double* val, const std::string& name = "", size_t cnt = 1) = 0;
        virtual ISaveVisitor& operator()(const void* binary, const std::string& name = "", size_t bytes = 1) = 0;

        template <class T>
        ISaveVisitor& operator()(const T* val, const std::string& name = "", size_t cnt = 1);

        template <class T>
        const T* getPointer(uint32_t id);

        virtual uint32_t getPointerId(TypeInfo type, const void* ptr) = 0;

        template <class T>
        uint32_t getPointerId(const T* ptr);

        template <class T>
        void setSerializedPointer(const T* ptr, uint32_t id);

      protected:
        virtual ISaveVisitor&
        operator()(const IStructTraits* trait, const void* inst, const std::string& name = "", size_t cnt = 1) = 0;
        virtual ISaveVisitor&
        operator()(const IContainerTraits* trait, const void* inst, const std::string& name = "", size_t cnt = 1) = 0;
        ISaveVisitor&
        saveTrait(const IStructTraits* trait, const void* inst, const std::string& name = "", size_t cnt = 1);
        ISaveVisitor&
        saveTrait(const IContainerTraits* trait, const void* inst, const std::string& name = "", size_t cnt = 1);
        virtual const void* getPointer(TypeInfo type, uint32_t id) = 0;

        virtual void setSerializedPointer(TypeInfo type, uint32_t id, const void* ptr) = 0;
    };

    struct MO_EXPORTS StaticVisitor
    {
        StaticVisitor() = default;
        StaticVisitor(const StaticVisitor&) = default;
        StaticVisitor(StaticVisitor&&) = default;
        StaticVisitor& operator=(const StaticVisitor&) = default;
        StaticVisitor& operator=(StaticVisitor&&) = default;

        virtual ~StaticVisitor();
        template <class T>
        void visit(const std::string& name, const size_t cnt = 1)
        {
            impl(name, cnt, static_cast<const T*>(nullptr));
        }

        virtual void visit(const ITraits*, const std::string& name, size_t cnt = 1) = 0;

      private:
        virtual void implDyn(TypeInfo, const std::string& name, size_t cnt) = 0;
        template <class T>
        auto impl(const std::string& name, const size_t cnt, const T*) -> ct::EnableIf<IsPrimitive<T>::value>;

        void impl(const std::string&, const size_t, const void*)
        {
        }

        template <class T>
        auto impl(const std::string& name, const size_t cnt, const T*)
            -> ct::EnableIf<!IsPrimitive<T>::value && !is_complete<TTraits<T>>::value>;

        template <class T>
        auto impl(const std::string& name, const size_t cnt, const T*)
            -> ct::EnableIf<!IsPrimitive<T>::value && is_complete<TTraits<T>>::value>;
    };

    ///////////////////////////////////////////////////////////////////////////////
    ///             IMPLEMENTATION
    ///////////////////////////////////////////////////////////////////////////////

    template <class T>
    struct Visit;

    template <class T>
    T* ILoadVisitor::getPointer(const uint32_t id)
    {
        void* ptr = getPointer(TypeInfo::create<T>(), id);
        return static_cast<T*>(ptr);
    }

    template <class T>
    void ILoadVisitor::setSerializedPointer(T* ptr, const uint32_t id)
    {
        setSerializedPointer(TypeInfo::create<T>(), id, ptr);
    }

    template <class T>
    const T* ISaveVisitor::getPointer(const uint32_t id)
    {
        const void* ptr = getPointer(TypeInfo::create<T>(), id);
        return static_cast<const T*>(ptr);
    }

    template <class T>
    uint32_t ISaveVisitor::getPointerId(const T* ptr)
    {
        return getPointerId(TypeInfo::create<T>(), ptr);
    }

    template <class T>
    void ISaveVisitor::setSerializedPointer(const T* ptr, const uint32_t id)
    {
        setSerializedPointer(TypeInfo::create<T>(), id, ptr);
    }

    template <class T>
    void IDynamicVisitor::pushCach(T&& val, const std::string& name, const uint64_t id)
    {
        auto& container = accessCache(name, id);
        container = std::unique_ptr<CacheDataContainer>(new DataContainer<T>(std::move(val)));
    }

    template <class T>
    bool IDynamicVisitor::tryPopCache(T& val, const std::string& name, const uint64_t id)
    {
        auto& container = accessCache(name, id);
        if (container)
        {
            auto typed = dynamic_cast<DataContainer<T>*>(container.get());
            if (typed)
            {
                val = typed->m_val;
                return true;
            }
        }
        return false;
    }

    template <class T>
    T IDynamicVisitor::popCache(const std::string& name, const uint64_t id)
    {
        T ret;
        if (!tryPopCache<T>(ret, name, id))
        {
            throw std::runtime_error(name + " not a valid cache entry");
        }
        return ret;
    }

    template <class T>
    TTraits<typename std::decay<T>::type> makeTraits(T*)
    {
        return {};
    }

    template <class T>
    ILoadVisitor& ILoadVisitor::operator()(T* val, const std::string& name, const size_t cnt)
    {
        auto traits = makeTraits(val);
        using base = typename decltype(traits)::base;
        loadTrait(static_cast<base*>(&traits), val, name, cnt);
        return *this;
    }

    template <class T>
    ISaveVisitor& ISaveVisitor::operator()(const T* val, const std::string& name, const size_t cnt)
    {
        auto traits = makeTraits(val);
        using base = typename decltype(traits)::base;
        saveTrait(static_cast<base*>(&traits), val, name, cnt);
        return *this;
    }

    template <class T>
    auto StaticVisitor::impl(const std::string& name, const size_t cnt, const T*) -> ct::EnableIf<IsPrimitive<T>::value>
    {
        implDyn(TypeInfo::create<T>(), name, cnt);
    }

    template <class T>
    auto StaticVisitor::impl(const std::string& name, const size_t cnt, const T*)
        -> ct::EnableIf<!IsPrimitive<T>::value && !is_complete<TTraits<T>>::value>
    {
        implDyn(TypeInfo::create<T>(), name, cnt);
        Visit<T>::visit(*this, name, cnt);
    }

    template <class T>
    auto StaticVisitor::impl(const std::string& name, const size_t cnt, const T*)
        -> ct::EnableIf<!IsPrimitive<T>::value && is_complete<TTraits<T>>::value>
    {
        const TTraits<T> trait;
        visit(static_cast<const ITraits*>(&trait), name, cnt);
    }
} // namespace mo

#endif // MO_VISITATION_IDYNAMICVISITOR_HPP
