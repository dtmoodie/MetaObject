#pragma once
#include "MetaObject/logging/logging.hpp"
#include <type_traits>
namespace mo
{
    template <class T1, class T2>
    struct LargerSize
    {
        static const bool value = sizeof(T1) > sizeof(T2);
    };
    template <class Type, class Enable>
    struct ParamTraitsImpl;

    /*!
     *  Specialization for POD types larger than a pointer, pass data around as a reference to data
     */
    template <class Type>
    struct ParamTraitsImpl<Type,
                           typename std::enable_if<std::is_pod<Type>::value && LargerSize<Type, void*>::value>::type>
    {
        enum
        {
            REQUIRES_GPU_SYNC = 0,
            HAS_TRIVIAL_MOVE = 1
        };
        typedef Type Raw_t;
        typedef Type Storage_t;
        typedef Type& TypeRef_t;
        typedef const Type& ConstTypeRef_t;
        typedef Type* StoragePtr_t;
        typedef const Type* ConstStoragePtr_t;
        typedef const Type& ConstStorageRef_t;
        typedef Type InputStorage_t;
        typedef const Type* Input_t;

        static inline Storage_t copy(const Type& value) { return value; }

        static inline Storage_t clone(const Type& value) { return value; }

        template <class... Args>
        static Type& reset(Storage_t& input_storage, Args&&... args)
        {
            input_storage = Type(std::forward<Args>(args)...);
            return input_storage;
        }
        static void move(Storage_t& storage, Type&& data) { storage = std::move(data); }

        template <class... Args>
        static void nullify(InputStorage_t& input_storage)
        {
            input_storage.reset();
        }
        static inline Type& get(Storage_t& value) { return value; }
        static inline const Type& get(ConstStorageRef_t value) { return value; }
        static inline Type* ptr(Storage_t& value) { return &value; }
        static inline const Type* ptr(ConstStorageRef_t value) { return &value; }
    };

    /*!
     *  Specialization for POD types that are smaller than the size of a pointer. Store data as a Type, pass data around
     * as a raw type
     */
    template <class Type>
    struct ParamTraitsImpl<Type,
                           typename std::enable_if<std::is_pod<Type>::value && !LargerSize<Type, void*>::value>::type>
    {
        enum
        {
            REQUIRES_GPU_SYNC = 0,
            HAS_TRIVIAL_MOVE = 1
        };
        typedef Type Raw_t;
        typedef Type Storage_t;
        typedef Type& TypeRef_t;
        typedef const Type& ConstTypeRef_t;
        typedef Type* StoragePtr_t;
        typedef const Type* ConstStoragePtr_t;
        typedef Type ConstStorageRef_t;
        typedef Type InputStorage_t;
        typedef const Type* Input_t;

        static inline Storage_t copy(const Type& value) { return value; }
        static inline Storage_t clone(const Type& value) { return value; }
        static void move(Storage_t& storage, Type&& data) { storage = std::move(data); }
        static void move(Storage_t& storage, Type& data) { storage = data; }
        template <class... Args>
        static Type& reset(Storage_t& input_storage, Args&&... args)
        {
            input_storage = Type(std::forward<Args>(args)...);
            return input_storage;
        }

        template <class... Args>
        static void nullify(InputStorage_t& input_storage)
        {
            input_storage.reset();
        }
        static inline Type& get(Storage_t& value) { return value; }
        static inline const Type& get(const Storage_t& value) { return value; }
        static inline Type* ptr(Type& value) { return &value; }
        static inline const Type* ptr(const Type& value) { return &value; }
    };
}
