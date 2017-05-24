#pragma once
#include "MetaObject/logging/Log.hpp"
#include <type_traits>

// forward declaration of weak ptrs
namespace rcc{
    template<class T> class weak_ptr;
    template<class T> class shared_ptr;
}
namespace std{
    template<class T> class shared_ptr;
}

namespace mo{
/*!
 *  Specialization for rcc::shared_ptr
 */
template<class Type> struct RccParamTraitsImplShared{
    enum {
        REQUIRES_GPU_SYNC = 0,
        HAS_TRIVIAL_MOVE = 0
    };
    typedef rcc::shared_ptr<Type> Raw_t;
    typedef rcc::shared_ptr<Type> Storage_t;
    typedef rcc::shared_ptr<Type>& TypeRef_t;
    typedef const Type& ConstTypeRef_t;
    typedef Type* StoragePtr_t;
    typedef const Type* ConstStoragePtr_t;
    typedef const rcc::shared_ptr<const Type>& ConstStorageRef_t;
    typedef rcc::shared_ptr<const Type> InputStorage_t;
    typedef const Type* Input_t;


    static inline Storage_t copy(const Storage_t& value) {
        return Storage_t(value);
    }
    // TODO figure out deep copy, perhaps create a new object from the constructor
    // Then use an ISimpleSerializer to copy members?
    static inline Storage_t clone(const Storage_t& value){
        return Storage_t();
    }

    template<class...Args>
    static Type& reset(Storage_t& input_storage, Args...args) {
        input_storage = Storage_t(std::forward<Args...>(args)...);
        return get(input_storage);
    }

    template<class...Args>
    static void reset(InputStorage_t& input_storage, Args...args) {
        //input_storage = Type(std::forward(args)...);
    }
    template<class...Args>
    static void nullify(InputStorage_t& input_storage) {
        input_storage.Reset();
    }
    static inline rcc::shared_ptr<Type>& get(Storage_t& value){
        MO_ASSERT(value);
        return value;
    }
    static inline const rcc::shared_ptr<const Type>& get(const InputStorage_t& value){
        MO_ASSERT(value);
        return value;
    }

    static inline Type* ptr(Storage_t& value){
        MO_ASSERT(value);
        return value.get();
    }
    static inline const Type* ptr(const InputStorage_t& value){
        MO_ASSERT(value);
        return value.get();
    }
}; // RccParamTraitsImplShared

/*!
 *  Specialization for rcc::weak_ptr
 */
template<class Type> struct RccParamTraitsImplWeak{
    enum {
        REQUIRES_GPU_SYNC = 0,
        HAS_TRIVIAL_MOVE = 0
    };
    typedef rcc::weak_ptr<Type> Raw_t;
    typedef rcc::weak_ptr<Type> Storage_t;
    typedef rcc::weak_ptr<Type>& TypeRef_t;
    typedef const Type& ConstTypeRef_t;
    typedef Type* StoragePtr_t;
    typedef const Type* ConstStoragePtr_t;
    typedef const rcc::weak_ptr<const Type>& ConstStorageRef_t;
    typedef rcc::weak_ptr<const Type> InputStorage_t;
    typedef const Type* Input_t;


    static inline Storage_t copy(const Storage_t& value) {
        return Storage_t(value);
    }
    // TODO figure out deep copy, perhaps create a new object from the constructor
    // Then use an ISimpleSerializer to copy members?
    static inline Storage_t clone(const Storage_t& value){
        return Storage_t();
    }

    template<class...Args>
    static Type& reset(Storage_t& input_storage, Args...args) {
        input_storage = Storage_t(std::forward<Args...>(args)...);
        return get(input_storage);
    }

    template<class...Args>
    static void reset(InputStorage_t& input_storage, Args...args) {
        //input_storage = Type(std::forward(args)...);
    }
    template<class...Args>
    static void nullify(InputStorage_t& input_storage) {
        input_storage.Reset();
    }
    static inline rcc::weak_ptr<Type>& get(Storage_t& value){
        return value;
    }
    static inline const rcc::weak_ptr<const Type>& get(const InputStorage_t& value){
        return value;
    }

    static inline Type* ptr(Storage_t& value){
        MO_ASSERT(value);
        return value.get();
    }
    static inline const Type* ptr(const InputStorage_t& value){
        MO_ASSERT(value);
        return value.get();
    }
}; // RccParamTraitsImplWeak

/*!
 *  Specialization for std::shared_ptr
 */
template<class Type> struct ParamTraitsShared{
    enum {
        REQUIRES_GPU_SYNC = 0,
        HAS_TRIVIAL_MOVE = 0
    };
    typedef std::shared_ptr<Type> Raw_t;
    typedef std::shared_ptr<Type> Storage_t;
    typedef Type& TypeRef_t;
    typedef const Type& ConstTypeRef_t;
    typedef Type* StoragePtr_t;
    typedef const Type* ConstStoragePtr_t;
    typedef const rcc::shared_ptr<const Type>& ConstStorageRef_t;
    typedef std::shared_ptr<const Type> InputStorage_t;
    typedef const Type* Input_t;


    static inline Storage_t copy(const Storage_t& value) {
        return Storage_t(value);
    }
    // TODO figure out deep copy, perhaps create a new object from the constructor
    // Then use an ISimpleSerializer to copy members?
    static inline Storage_t clone(const Storage_t& value){
        return Storage_t();
    }

    template<class...Args>
    static Type& reset(Storage_t& input_storage, Args...args) {
        input_storage = Storage_t(std::forward<Args...>(args)...);
        return get(input_storage);
    }

    template<class...Args>
    static void reset(InputStorage_t& input_storage, Args...args) {
        //input_storage = Type(std::forward(args)...);
    }
    template<class...Args>
    static void nullify(InputStorage_t& input_storage) {
        input_storage.Reset();
    }
    static inline Type& get(Storage_t& value){
        MO_ASSERT(value);
        return *value.get();
    }
    static inline const Type& get(const InputStorage_t& value){
        MO_ASSERT(value);
        return *value.get();
    }

    static inline Type* ptr(Storage_t& value){
        MO_ASSERT(value);
        return value.get();
    }
    static inline const Type* ptr(const InputStorage_t& value){
        MO_ASSERT(value);
        return value.get();
    }
};



} // namespace mo
