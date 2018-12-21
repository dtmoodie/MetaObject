#pragma once
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

    struct IDynamicVisitor;
    struct IReadVisitor;
    struct IWriteVisitor;
    struct ITraits
    {
        virtual ~ITraits() = default;
        virtual void visit(IReadVisitor* visitor) = 0;
        virtual void visit(IWriteVisitor* visitor) const = 0;
        virtual TypeInfo type() const = 0;
        virtual std::string getName() const = 0;
    };

    struct IStructTraits : public ITraits
    {
        // sizeof(T)
        virtual size_t size() const = 0;
        // can be serialized via a memcpy(ptr)
        virtual bool triviallySerializable() const = 0;
        // if it can be serialized by one of the primitive supported types, such as
        // struct{float x,y,z;} can be serialized as 3 floats in continuous memory
        virtual bool isPrimitiveType() const = 0;
        // const ptr to type
        virtual const void* ptr() const = 0;
        // non const ptr to type
        virtual void* ptr() = 0;
    };

    struct IContainerTraits : public ITraits
    {
        virtual TypeInfo keyType() const = 0;
        virtual TypeInfo valueType() const = 0;

        virtual bool isContinuous() const = 0;
        virtual bool podValues() const = 0;
        virtual bool podKeys() const = 0;
        virtual size_t getSize() const = 0;
        virtual void setSize(const size_t num) = 0;
    };
    template <class T>
    struct ArrayContainerTrait;

    template <class T, class E = void>
    struct TTraits;

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

    template <class T, class E = void>
    struct IsPrimitive : public std::false_type
    {
    };

    template <class T>
    struct IsPrimitive<T,
                       typename std::enable_if<std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value ||
                                               std::is_same<T, int16_t>::value || std::is_same<T, uint16_t>::value ||
                                               std::is_same<T, int32_t>::value || std::is_same<T, uint32_t>::value ||
                                               std::is_same<T, int64_t>::value || std::is_same<T, uint64_t>::value ||
                                               std::is_same<T, float>::value || std::is_same<T, double>::value ||
                                               std::is_same<T, void>::value>::type> : public std::true_type
    {
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

    struct IReadVisitor : public virtual IDynamicVisitor
    {
        virtual IReadVisitor& operator()(bool* val, const std::string& name = "", const size_t cnt = 1) = 0;
        virtual IReadVisitor& operator()(char* val, const std::string& name = "", const size_t cnt = 1) = 0;
        virtual IReadVisitor& operator()(int8_t* val, const std::string& name = "", const size_t cnt = 1) = 0;
        virtual IReadVisitor& operator()(uint8_t* val, const std::string& name = "", const size_t cnt = 1) = 0;
        virtual IReadVisitor& operator()(int16_t* val, const std::string& name = "", const size_t cnt = 1) = 0;
        virtual IReadVisitor& operator()(uint16_t* val, const std::string& name = "", const size_t cnt = 1) = 0;
        virtual IReadVisitor& operator()(int32_t* val, const std::string& name = "", const size_t cnt = 1) = 0;
        virtual IReadVisitor& operator()(uint32_t* val, const std::string& name = "", const size_t cnt = 1) = 0;
        virtual IReadVisitor& operator()(int64_t* val, const std::string& name = "", const size_t cnt = 1) = 0;
        virtual IReadVisitor& operator()(uint64_t* val, const std::string& name = "", const size_t cnt = 1) = 0;

        virtual IReadVisitor& operator()(long long* val, const std::string& name = "", const size_t cnt = 1) = 0;
        virtual IReadVisitor& operator()(unsigned long long* val, const std::string& name = "", const size_t cnt = 1) = 0;

        virtual IReadVisitor& operator()(float* val, const std::string& name = "", const size_t cnt = 1) = 0;
        virtual IReadVisitor& operator()(double* val, const std::string& name = "", const size_t cnt = 1) = 0;
        virtual IReadVisitor& operator()(void* binary, const std::string& name = "", const size_t num_bytes = 1) = 0;

        template <class T>
        auto operator()(T* val, const std::string& name = "", const size_t cnt = 1)
            -> enable_if_trait_exists<T, IReadVisitor&>;
        template <class T>
        auto operator()(T* val, const std::string& name = "", const size_t cnt = 1)
            -> enable_if_not_trait_exists<T, IReadVisitor&>;

        virtual IReadVisitor& operator()(IStructTraits* val, const std::string& name = "") = 0;
        virtual IReadVisitor& operator()(IContainerTraits* val, const std::string& name = "") = 0;

        template <class T>
        T* getPointer(const uint64_t id);

        template <class T>
        void setSerializedPointer(T* ptr, const uint64_t id);

        virtual std::string getCurrentElementName() const = 0;

      protected:
        virtual void* getPointer(const TypeInfo type, const uint64_t id) = 0;
        virtual void setSerializedPointer(const TypeInfo type, const uint64_t id, void* ptr) = 0;
    };

    struct IWriteVisitor : public virtual IDynamicVisitor
    {
        virtual IWriteVisitor& operator()(const bool* val, const std::string& name = "", const size_t cnt = 1) = 0;
        virtual IWriteVisitor& operator()(const char* val, const std::string& name = "", const size_t cnt = 1) = 0;
        virtual IWriteVisitor& operator()(const int8_t* val, const std::string& name = "", const size_t cnt = 1) = 0;
        virtual IWriteVisitor& operator()(const uint8_t* val, const std::string& name = "", const size_t cnt = 1) = 0;
        virtual IWriteVisitor& operator()(const int16_t* val, const std::string& name = "", const size_t cnt = 1) = 0;
        virtual IWriteVisitor& operator()(const uint16_t* val, const std::string& name = "", const size_t cnt = 1) = 0;
        virtual IWriteVisitor& operator()(const int32_t* val, const std::string& name = "", const size_t cnt = 1) = 0;
        virtual IWriteVisitor& operator()(const uint32_t* val, const std::string& name = "", const size_t cnt = 1) = 0;
        virtual IWriteVisitor& operator()(const int64_t* val, const std::string& name = "", const size_t cnt = 1) = 0;
        virtual IWriteVisitor& operator()(const uint64_t* val, const std::string& name = "", const size_t cnt = 1) = 0;

        virtual IWriteVisitor& operator()(const long long* val, const std::string& name = "", const size_t cnt = 1) = 0;
        virtual IWriteVisitor& operator()(const unsigned long long* val, const std::string& name = "", const size_t cnt = 1) = 0;
        virtual IWriteVisitor& operator()(const float* val, const std::string& name = "", const size_t cnt = 1) = 0;
        virtual IWriteVisitor& operator()(const double* val, const std::string& name = "", const size_t cnt = 1) = 0;
        virtual IWriteVisitor& operator()(const void* binary, const std::string& name = "", const size_t bytes = 1) = 0;

        template <class T>
        auto operator()(const T* val, const std::string& name = "", const size_t cnt = 1)
            -> enable_if_trait_exists<T, IWriteVisitor&>;

        template <class T>
        auto operator()(const T* val, const std::string& name = "", const size_t cnt = 1)
            -> enable_if_not_trait_exists<T, IWriteVisitor&>;

        virtual IWriteVisitor& operator()(const IStructTraits* val, const std::string& name = "") = 0;
        virtual IWriteVisitor& operator()(const IContainerTraits* val, const std::string& name = "") = 0;

        template <class T>
        const T* getPointer(const uint64_t id);

        template <class T>
        void setSerializedPointer(const T* ptr, const uint64_t id);

      protected:
        virtual const void* getPointer(const TypeInfo type, const uint64_t id) = 0;
        virtual void setSerializedPointer(const TypeInfo type, const uint64_t id, const void* ptr) = 0;
    };

    template <class T>
    T* IReadVisitor::getPointer(const uint64_t id)
    {
        void* ptr = getPointer(TypeInfo(typeid(T)), id);
        return static_cast<T*>(ptr);
    }

    template <class T>
    void IReadVisitor::setSerializedPointer(T* ptr, const uint64_t id)
    {
        setSerializedPointer(TypeInfo(typeid(T)), id, ptr);
    }

    template <class T>
    const T* IWriteVisitor::getPointer(const uint64_t id)
    {
        const void* ptr = getPointer(TypeInfo(typeid(T)), id);
        return static_cast<const T*>(ptr);
    }

    template <class T>
    void IWriteVisitor::setSerializedPointer(const T* ptr, const uint64_t id)
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
    TTraits<T> makeTraits(T* const val)
    {
        return TTraits<T>(val, nullptr);
    }

    template <class T>
    TTraits<T> makeTraits(const T* val)
    {
        return TTraits<T>(nullptr, val);
    }

    template <class T>
    auto IReadVisitor::operator()(T* val, const std::string& name, const size_t cnt)
        -> enable_if_trait_exists<T, IReadVisitor&>
    {
        if (cnt == 1)
        {
            auto traits = makeTraits(val);
            using base = typename decltype(traits)::base;
            (*this)(static_cast<base*>(&traits), name);
        }
        else
        {
            ArrayContainerTrait<T> traits(val, nullptr, cnt);
            using base = typename decltype(traits)::base;
            (*this)(static_cast<base*>(&traits), name);
        }
        return *this;
    }

    template <class T>
    auto IReadVisitor::operator()(T* val, const std::string& name, const size_t cnt)
        -> enable_if_not_trait_exists<T, IReadVisitor&>
    {
        return read(*this, val, name, cnt);
    }

    template <class T>
    auto IWriteVisitor::operator()(const T* val, const std::string& name, const size_t cnt)
        -> enable_if_trait_exists<T, IWriteVisitor&>
    {
        if (cnt == 1)
        {
            auto traits = makeTraits(val);
            using base = typename decltype(traits)::base;
            (*this)(static_cast<base*>(&traits), name);
        }
        else
        {
            ArrayContainerTrait<T> traits(nullptr, val, cnt);
            using base = typename decltype(traits)::base;
            (*this)(static_cast<base*>(&traits), name);
        }
        return *this;
    }

    // In this case, T is some kind of struct that we do not have a specialization for
    //template <class T>
    //IWriteVisitor& visit(IWriteVisitor& visitor, const T* val, const std::string& name, const size_t cnt);

    template <class T>
    auto IWriteVisitor::operator()(const T* val, const std::string& name, const size_t cnt)
        -> enable_if_not_trait_exists<T, IWriteVisitor&>
    {
        return write(*this, val, name, cnt);
    }
}
#include "VisitorTraits.hpp"
#include "visitor_traits/array.hpp"