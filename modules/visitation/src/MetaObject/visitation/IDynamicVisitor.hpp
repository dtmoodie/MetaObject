#ifndef MO_VISITATION_IDYNAMICVISITOR_HPP
#define MO_VISITATION_IDYNAMICVISITOR_HPP

#include "TraitInterface.hpp"
#include "VisitorTraits.hpp"
#include "visitor_traits/array.hpp"
#include <MetaObject/detail/TypeInfo.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>

namespace mo
{
    struct CacheDataContainer
    {
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
    using enable_if_trait_exists = typename std::enable_if<is_complete<TTraits<T>>::value, U>::type;

    template <class T, class U = void>
    using enable_if_not_trait_exists = typename std::enable_if<!is_complete<TTraits<T>>::value, U>::type;

    template <class T>
    struct IsPrimitive
    {
        using type = typename std::remove_cv<T>::type;
        static constexpr const bool value =
            std::is_same<type, int8_t>::value || std::is_same<type, uint8_t>::value ||
            std::is_same<type, int16_t>::value || std::is_same<type, uint16_t>::value ||
            std::is_same<type, int32_t>::value || std::is_same<type, uint32_t>::value ||
            std::is_same<type, int64_t>::value || std::is_same<type, uint64_t>::value ||
            std::is_same<type, long long>::value || std::is_same<type, unsigned long long>::value ||
            std::is_same<type, float>::value || std::is_same<type, double>::value || std::is_same<type, void*>::value ||
            std::is_same<type, char>::value || std::is_same<type, bool>::value;
    };

    struct VisitorTraits
    {
        // if true, the name field of the () operator is used to search for the provided data
        bool supports_named_access;
        // If this is true, read data from external source and put into the visited struct
        // else read data from struct and put into output
        bool reader;
    };

    struct IDynamicVisitor
    {
        virtual ~IDynamicVisitor();

        virtual VisitorTraits traits() const = 0;

        template <class T>
        void pushCach(T&& val, const std::string& name, const uint64_t id = 0);

        template <class T>
        bool tryPopCache(T& val, const std::string& name, const uint64_t id = 0);

        template <class T>
        T popCache(const std::string& name, const uint64_t id = 0);

      protected:
        virtual std::unique_ptr<CacheDataContainer>& accessCache(const std::string& name, const uint64_t id = 0) = 0;
    };

    struct ILoadVisitor : public virtual IDynamicVisitor
    {
        virtual ILoadVisitor& operator()(char* val, const std::string& name = "", const size_t cnt = 1) = 0;
        virtual ILoadVisitor& operator()(bool* val, const std::string& name = "", const size_t cnt = 1) = 0;
        virtual ILoadVisitor& operator()(int8_t* val, const std::string& name = "", const size_t cnt = 1) = 0;
        virtual ILoadVisitor& operator()(uint8_t* val, const std::string& name = "", const size_t cnt = 1) = 0;
        virtual ILoadVisitor& operator()(int16_t* val, const std::string& name = "", const size_t cnt = 1) = 0;
        virtual ILoadVisitor& operator()(uint16_t* val, const std::string& name = "", const size_t cnt = 1) = 0;
        virtual ILoadVisitor& operator()(int32_t* val, const std::string& name = "", const size_t cnt = 1) = 0;
        virtual ILoadVisitor& operator()(uint32_t* val, const std::string& name = "", const size_t cnt = 1) = 0;
        virtual ILoadVisitor& operator()(int64_t* val, const std::string& name = "", const size_t cnt = 1) = 0;
        virtual ILoadVisitor& operator()(uint64_t* val, const std::string& name = "", const size_t cnt = 1) = 0;

        virtual ILoadVisitor& operator()(long long* val, const std::string& name = "", const size_t cnt = 1) = 0;
        virtual ILoadVisitor&
        operator()(unsigned long long* val, const std::string& name = "", const size_t cnt = 1) = 0;

        virtual ILoadVisitor& operator()(float* val, const std::string& name = "", const size_t cnt = 1) = 0;
        virtual ILoadVisitor& operator()(double* val, const std::string& name = "", const size_t cnt = 1) = 0;
        virtual ILoadVisitor& operator()(void* binary, const std::string& name = "", const size_t num_bytes = 1) = 0;

        template <class T>
        auto operator()(T* val, const std::string& name = "", const size_t cnt = 1)
            -> enable_if_trait_exists<T, ILoadVisitor&>;
        template <class T>
        auto operator()(T* val, const std::string& name = "", const size_t cnt = 1)
            -> enable_if_not_trait_exists<T, ILoadVisitor&>;

        virtual ILoadVisitor& operator()(ILoadStructTraits* val, const std::string& name = "") = 0;
        virtual ILoadVisitor& operator()(ILoadContainerTraits* val, const std::string& name = "") = 0;

        template <class T>
        T* getPointer(const uint64_t id);

        template <class T>
        void setSerializedPointer(T* ptr, const uint64_t id);

        virtual std::string getCurrentElementName() const = 0;

      protected:
        ILoadVisitor& loadTrait(ILoadStructTraits* val, const std::string& name = "")
        {
            return (*this)(val, name);
        }
        ILoadVisitor& loadTrait(ILoadContainerTraits* val, const std::string& name = "")
        {
            return (*this)(val, name);
        }

        virtual void* getPointer(const TypeInfo type, const uint64_t id) = 0;
        virtual void setSerializedPointer(const TypeInfo type, const uint64_t id, void* ptr) = 0;
    };

    struct ISaveVisitor : public virtual IDynamicVisitor
    {
        virtual ISaveVisitor& operator()(const bool* val, const std::string& name = "", const size_t cnt = 1) = 0;
        virtual ISaveVisitor& operator()(const char* val, const std::string& name = "", const size_t cnt = 1) = 0;
        virtual ISaveVisitor& operator()(const int8_t* val, const std::string& name = "", const size_t cnt = 1) = 0;
        virtual ISaveVisitor& operator()(const uint8_t* val, const std::string& name = "", const size_t cnt = 1) = 0;
        virtual ISaveVisitor& operator()(const int16_t* val, const std::string& name = "", const size_t cnt = 1) = 0;
        virtual ISaveVisitor& operator()(const uint16_t* val, const std::string& name = "", const size_t cnt = 1) = 0;
        virtual ISaveVisitor& operator()(const int32_t* val, const std::string& name = "", const size_t cnt = 1) = 0;
        virtual ISaveVisitor& operator()(const uint32_t* val, const std::string& name = "", const size_t cnt = 1) = 0;
        virtual ISaveVisitor& operator()(const int64_t* val, const std::string& name = "", const size_t cnt = 1) = 0;
        virtual ISaveVisitor& operator()(const uint64_t* val, const std::string& name = "", const size_t cnt = 1) = 0;

        virtual ISaveVisitor& operator()(const long long* val, const std::string& name = "", const size_t cnt = 1) = 0;
        virtual ISaveVisitor&
        operator()(const unsigned long long* val, const std::string& name = "", const size_t cnt = 1) = 0;
        virtual ISaveVisitor& operator()(const float* val, const std::string& name = "", const size_t cnt = 1) = 0;
        virtual ISaveVisitor& operator()(const double* val, const std::string& name = "", const size_t cnt = 1) = 0;
        virtual ISaveVisitor& operator()(const void* binary, const std::string& name = "", const size_t bytes = 1) = 0;

        template <class T>
        auto operator()(const T* val, const std::string& name = "", const size_t cnt = 1)
            -> enable_if_trait_exists<T, ISaveVisitor&>;

        template <class T>
        auto operator()(const T* val, const std::string& name = "", const size_t cnt = 1)
            -> enable_if_not_trait_exists<T, ISaveVisitor&>;

        virtual ISaveVisitor& operator()(ISaveStructTraits* val, const std::string& name = "") = 0;
        virtual ISaveVisitor& operator()(ISaveContainerTraits* val, const std::string& name = "") = 0;

        template <class T>
        const T* getPointer(const uint64_t id);

        template <class T>
        void setSerializedPointer(const T* ptr, const uint64_t id);

      protected:
        virtual const void* getPointer(const TypeInfo type, const uint64_t id) = 0;
        virtual void setSerializedPointer(const TypeInfo type, const uint64_t id, const void* ptr) = 0;
    };

    struct StaticVisitor
    {
        virtual ~StaticVisitor();
        template <class T>
        void visit(const std::string& name, const size_t cnt = 1)
        {
            impl(name, cnt, static_cast<const T*>(nullptr));
        }

        virtual void visit(const ITraits*, const std::string& name, const size_t cnt = 1) = 0;

      private:
        virtual void implDyn(const TypeInfo, const std::string& name, const size_t cnt) = 0;
        template <class T>
        auto impl(const std::string& name, const size_t cnt, const T*) ->
            typename std::enable_if<IsPrimitive<T>::value>::type;

        void impl(const std::string&, const size_t, const void*)
        {
        }

        template <class T>
        auto impl(const std::string& name, const size_t cnt, const T*) ->
            typename std::enable_if<!IsPrimitive<T>::value && !is_complete<TTraits<T>>::value>::type;

        template <class T>
        auto impl(const std::string& name, const size_t cnt, const T*) ->
            typename std::enable_if<!IsPrimitive<T>::value && is_complete<TTraits<T>>::value>::type;
    };

    template <class T>
    struct Visit;

    template <class T>
    T* ILoadVisitor::getPointer(const uint64_t id)
    {
        void* ptr = getPointer(TypeInfo(typeid(T)), id);
        return static_cast<T*>(ptr);
    }

    template <class T>
    void ILoadVisitor::setSerializedPointer(T* ptr, const uint64_t id)
    {
        setSerializedPointer(TypeInfo(typeid(T)), id, ptr);
    }

    template <class T>
    const T* ISaveVisitor::getPointer(const uint64_t id)
    {
        const void* ptr = getPointer(TypeInfo(typeid(T)), id);
        return static_cast<const T*>(ptr);
    }

    template <class T>
    void ISaveVisitor::setSerializedPointer(const T* ptr, const uint64_t id)
    {
        setSerializedPointer(TypeInfo(typeid(T)), id, ptr);
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
    auto makeTraits(T* val, const size_t cnt = 1) ->
        typename std::enable_if<std::is_base_of<IStructTraits, TTraits<T>>::value, TTraits<T>>::type
    {
        return TTraits<T>(val, cnt);
    }

    template <class T>
    auto makeTraits(const T* val, const size_t cnt = 1) ->
        typename std::enable_if<std::is_base_of<IStructTraits, TTraits<const T>>::value, TTraits<const T>>::type
    {
        return TTraits<const T>(val, cnt);
    }

    template <class T>
    auto makeTraits(T* val, const size_t = 1) ->
        typename std::enable_if<std::is_base_of<IContainerTraits, TTraits<T>>::value, TTraits<T>>::type
    {
        return TTraits<T>(val);
    }

    template <class T>
    auto makeTraits(const T* val, const size_t = 1) ->
        typename std::enable_if<std::is_base_of<IContainerTraits, TTraits<const T>>::value, TTraits<const T>>::type
    {
        return TTraits<const T>(val);
    }

    template <class T>
    auto ILoadVisitor::operator()(T* val, const std::string& name, const size_t cnt)
        -> enable_if_trait_exists<T, ILoadVisitor&>
    {
        auto traits = makeTraits(val, cnt);
        loadTrait(&traits, name);
        return *this;
    }

    template <class T>
    auto ILoadVisitor::operator()(T* val, const std::string& name, const size_t cnt)
        -> enable_if_not_trait_exists<T, ILoadVisitor&>
    {
        return Visit<T>::load(*this, val, name, cnt);
    }

    template <class T>
    auto ISaveVisitor::operator()(const T* val, const std::string& name, const size_t cnt)
        -> enable_if_trait_exists<T, ISaveVisitor&>
    {
        auto traits = makeTraits(val, cnt);
        using base = typename decltype(traits)::base;
        (*this)(static_cast<base*>(&traits), name);
        return *this;
    }

    // In this case, T is some kind of struct that we do not have a specialization for
    // template <class T>
    // ISaveVisitor& visit(ISaveVisitor& visitor, const T* val, const std::string& name, const size_t cnt);

    template <class T>
    auto ISaveVisitor::operator()(const T* val, const std::string& name, const size_t cnt)
        -> enable_if_not_trait_exists<T, ISaveVisitor&>
    {
        return Visit<T>::save(*this, val, name, cnt);
    }

    template <class T>
    auto StaticVisitor::impl(const std::string& name, const size_t cnt, const T*) ->
        typename std::enable_if<IsPrimitive<T>::value>::type
    {
        implDyn(TypeInfo(typeid(T)), name, cnt);
    }

    template <class T>
    auto StaticVisitor::impl(const std::string& name, const size_t cnt, const T*) ->
        typename std::enable_if<!IsPrimitive<T>::value && !is_complete<TTraits<T>>::value>::type
    {
        implDyn(TypeInfo(typeid(T)), name, cnt);
        Visit<T>::visit(*this, name, cnt);
    }

    template <class T>
    auto StaticVisitor::impl(const std::string& name, const size_t cnt, const T*) ->
        typename std::enable_if<!IsPrimitive<T>::value && is_complete<TTraits<T>>::value>::type
    {
        const auto trait = makeTraits<T>(static_cast<const T*>(nullptr), cnt);
        visit(dynamic_cast<const ITraits*>(&trait), name, cnt);
    }
}

#endif // MO_VISITATION_IDYNAMICVISITOR_HPP
